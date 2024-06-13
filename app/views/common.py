from flask import flash, redirect, request, session, url_for, abort, jsonify
from flask_login import current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from app.custom import CustomBlueprint
from app.evaluator import pdf_to_text, separate_answers, calculate_similarity
from app.models import Invitation, User, db, USER_ROLES, Exam, Submission
from app.utils import logger, cache


from datetime import datetime
import json
import os


class CommonView(CustomBlueprint):
    def __init__(self):
        super().__init__("common", __name__, url_prefix="/")

        self.resticted_roles = ("admin", "teacher")
        self.resticted_routes = (
            "invite",
            "delete_invite",
            "exams",
            "change_exam_status",
            "delete_exam",
            "users",
            "delete_user",
            "add_user",
        )
        self.before_request(self.before_request_hook)

        # Authentication
        self.route("/", methods=["GET", "POST"])(self.login)
        self.route("/login", methods=["GET", "POST"])(self.login)
        self.route("/register", methods=["GET", "POST"])(self.signup)
        self.route("/logout")(self.logout)
        self.route("/guidelines")(self.guidelines)

        # Invitation management
        self.route("/invite", methods=["GET", "POST"])(self.invite)
        self.route("/delete-invite/<invitation_id>")(self.delete_invite)

        # Exam management
        self.route("/all-exams", methods=["GET", "POST"])(self.exams)
        self.route("/change-exam-status/<exam_id>")(self.change_exam_status)
        self.route("/delete-exam/<exam_id>")(self.delete_exam)

        # User management
        self.route("/all-<role>s")(self.users)
        self.route("/add-user", methods=["POST"])(self.add_user)
        self.route("/delete-user/<user_id>")(self.delete_user)

        # Submissions
        self.route("/history")(self.history)
        self.route("/submission", methods=["GET", "POST"])(self.submission)
        self.route("/result", methods=["GET", "POST"])(self.result)

    def before_request_hook(self):
        current_endpoint = request.endpoint.replace(f"{self.name}.", "")
        if (
            current_user
            and current_endpoint in self.resticted_routes
            and current_user.role not in self.resticted_roles
        ):
            return abort(403)

    def guidelines(self):
        return self.view("guidelines")

    def login(self):
        if current_user.is_authenticated:
            return redirect(url_for(f"{current_user.role}.index"))

        if request.method == "POST":
            email = request.form.get("email")
            password = request.form.get("password")

            user = User.query.filter_by(email=email).first()

            if user and check_password_hash(user.password_hash, password):
                login_user(user)
                flash("Logged in successfully.", "success")
            else:
                flash("Invalid email or password. Please try again.", "danger")
            return redirect(url_for("common.login"))

        return self.view("login")

    def signup(self):
        if current_user.is_authenticated:
            return redirect(url_for(f"{current_user.role}.index"))

        if request.method == "POST":
            name = request.form.get("name")
            email = request.form.get("email")
            password = request.form.get("password")
            invitation_code = request.form.get("invitation_code")

            existing_user = User.query.filter_by(email=email).first()

            if existing_user:
                flash(
                    "Email address already exists. Please choose a different one.",
                    "danger",
                )
                return redirect(url_for("common.signup"))

            invitation = Invitation.query.filter_by(
                code=invitation_code, email=email
            ).first()

            new_user = User(
                username=name,
                email=email,
                password_hash=generate_password_hash(password),
                role=invitation.role if invitation is not None else "teacher", 
                # By default role assign by user 
            )

            try:
                db.session.add(new_user)
                db.session.commit()

                if invitation is not None:
                    invitation.is_used = True
                    db.session.commit()

                login_user(new_user)
                flash("Your account has been created successfully.", "success")
                return redirect(url_for("common.login"))
            except Exception as e:
                db.session.rollback()
                logger.error(e)
                flash("An error occurred. Please try again.", "danger")

        return self.view("signup")

    @login_required
    def logout(self):
        logout_user()
        session.clear()
        return redirect(url_for("common.login"))

    @login_required
    def invite(self):
        if request.method == "POST":
            email = request.form.get("email")
            role = request.form.get("role")

            invitation = Invitation(email=email, role=role, invited_by=current_user.id)
            db.session.add(invitation)
            db.session.commit()

            flash(f"Use the code {invitation.code} to invite {email}.", "success")
            return redirect(url_for("common.invite"))

        all_roles = list(USER_ROLES)
        all_roles.remove("admin")

        if current_user.role in all_roles:
            all_roles.remove(current_user.role)

        invitations = Invitation.query.all()
        return self.view(
            "invite",
            required_roles=self.resticted_roles,
            all_roles=all_roles,
            invitations=invitations,
        )

    @login_required
    def delete_invite(self, invitation_id):
        invitation = Invitation.query.get_or_404(invitation_id)
        db.session.delete(invitation)
        db.session.commit()
        return redirect(url_for("common.invite"))

    @login_required
    def exams(self):
        if request.method == "POST":
            if 'file' not in request.files:
                return redirect(request.url)
            
            title = request.form.get("title")
            total_marks = request.form.get("total_marks")
            threshold = request.form.get("threshold")

            pdf_file = request.files.get("file")
            file_path = f"uploads/exams/{datetime.now().strftime('%Y%m%d%H%M%S')}-{pdf_file.filename}"
            pdf_file.save(f"app/static/{file_path}")

            answers_text = pdf_to_text(f"app/static/{file_path}")
            answers = separate_answers(answers_text)

            exam = Exam(
                title=title,
                answers=json.dumps(answers),
                file_path=file_path,
                total_marks=total_marks,
                threshold=threshold,
                teacher_id=current_user.id,
            )

            db.session.add(exam)
            db.session.commit()

            flash("Exam created successfully.", "success")
            return redirect(url_for("common.exams"))

        if current_user.role == "admin":
            exams = Exam.query.all()
        else:
            exams = Exam.query.filter_by(teacher_id=current_user.id).all()
        for exam in exams:
            exam.num_submissions = len(exam.submissions)
            exam.answer_obj = json.loads(exam.answers)
        return self.view(
            "exams", required_roles=self.resticted_roles, exams=exams, title="Exam"
        )

    @login_required
    def change_exam_status(self, exam_id):
        exam = Exam.query.get_or_404(exam_id)
        exam.is_finished = not exam.is_finished
        db.session.commit()
        return redirect(url_for("common.exams"))

    @login_required
    def delete_exam(self, exam_id):
        exam = Exam.query.get_or_404(exam_id)

        if exam.is_finished:
            flash("You cannot delete a finished exam.", "danger")
            return redirect(url_for("common.exams"))

        if exam.submissions:
            flash("You cannot delete an exam with submissions.", "danger")
            return redirect(url_for("common.exams"))

        if exam.teacher_id != current_user.id:
            flash("You cannot delete an exam you did not create.", "danger")
            return redirect(url_for("common.exams"))

        file_path = f"app/static/{exam.file_path}"
        if exam.file_path and os.path.exists(file_path):
            os.remove(file_path)

        db.session.delete(exam)
        db.session.commit()
        return redirect(url_for("common.exams"))

    @login_required
    def users(self, role=None):
        if role not in USER_ROLES or role == current_user.role:
            abort(404)

        users = User.query.filter_by(role=role).all()
        return self.view("users", required_roles=self.resticted_roles, title=role, users=users)

    @login_required
    def delete_user(self, user_id):
        user = User.query.get_or_404(user_id)
        db.session.delete(user)
        db.session.commit()
        return redirect(url_for("common.users", role=user.role))
    
    @login_required
    def add_user(self):
        name = request.form.get("name")
        email = request.form.get("email")
        role = request.form.get("role")

        existing_user = User.query.filter_by(email=email).first()

        if existing_user:
            flash(
                "Email address already exists. Please choose a different one.",
                "danger",
            )
            return redirect(url_for("common.users", role=role))

        new_user = User(
            username=name,
            email=email,
            password_hash=generate_password_hash("password"),
            role=role
        )

        try:
            db.session.add(new_user)
            db.session.commit()

            flash("User added successfully.", "success")
            return redirect(url_for("common.users", role=role))
        except Exception as e:
            db.session.rollback()
            logger.error(e)
            flash("An error occurred. Please try again.", "danger")

    @login_required
    def history(self):
        student_id = current_user.id if current_user.role == "student" else request.args.get('student')

        if student_id:
            student = User.query.get(student_id)
            submissions = Submission.query.filter_by(student_id=student_id).order_by(Submission.created_at.desc()).all()

            for submission in submissions:
                submission.submitted_at = submission.created_at.strftime("%A, %d %B %Y at %I:%M %p")
                threshold = submission.exam.threshold
                similarity_scores = json.loads(submission.similarity_scores)
                filtered_scores = [((score - threshold) / (1 - threshold)) if score >= threshold else 0 for score in similarity_scores.values()]
                overall_score = round(sum(filtered_scores) / len(filtered_scores), 2) if filtered_scores else 0
                submission.is_passed = overall_score >= 0.5
            return self.view("history", submissions=submissions, student=student, required_roles="all")
        
        users = User.query.filter_by(role="student").all()
        return self.view("history", users=users, required_roles="all")
        
    @login_required
    def submission(self):
        if request.method == 'POST':
            if 'file' not in request.files:
                return redirect(request.url)
            
            exam_id = request.form.get("exam")
            student_id = request.form.get("student")

            exam = Exam.query.get_or_404(exam_id)
            teacher_answers = json.loads(exam.answers)

            pdf_file = request.files.get("file")
            file_path = f"uploads/submissions/{datetime.now().strftime('%Y%m%d%H%M%S')}-{pdf_file.filename}"
            pdf_file.save(f"app/static/{file_path}")

            student_answers_text = pdf_to_text(f"app/static/{file_path}")
            print(student_answers_text)
            student_answers = separate_answers(student_answers_text)
            print(student_answers)

            similarity_scores = calculate_similarity(teacher_answers, student_answers)
            similarity_scores = {str(k): float(v) for k, v in similarity_scores.items()}

            try:
                submission = Submission(
                    exam_id=exam_id,
                    student_id=student_id,
                    file_path=file_path,
                    submitted_answers=json.dumps(student_answers),
                    similarity_scores=json.dumps(similarity_scores)
                )
                db.session.add(submission)
                db.session.commit()
                return redirect(url_for("common.result", submission=submission.id))
            except Exception as e:
                flash("Something went wrong. Please try again later.", "danger")
                return redirect(url_for("common.submission"))
            
        
        exams = Exam.query.filter_by(is_finished=False).all()
        students = User.query.filter_by(role="student").all()
        for exam in exams:
            exam.answer_obj = json.loads(exam.answers)
        return self.view("submission", exams=exams, students=students, required_roles="all")
    
    @login_required
    def result(self):
        if request.method == 'POST':
            
            submission_id = request.json.get("submission_id")
            submission = Submission.query.get(submission_id)
            threshold = submission.exam.threshold

            if submission is None:
                return jsonify({"error": "Submission not found."}), 404

            similarity_scores = json.loads(submission.similarity_scores)

            results = {}
            for student_key, similarity_score in similarity_scores.items():
                final_score = (similarity_score - threshold) / (1 - threshold) if similarity_score >= threshold else 0
                results[student_key] = final_score

            exam_dict = submission.exam.__dict__
            exam_dict.pop('_sa_instance_state', None)

            submission_dict = submission.__dict__
            submission_dict.pop('_sa_instance_state', None)
            submission_dict.pop('exam', None)

            submission_dict['created_at'] = submission.created_at.strftime("%a, %d %B %Y<br>%I:%M %p")

            filtered_scores = [score for score in results.values() if score > 0]
            overall_score = sum(filtered_scores) / len(filtered_scores) if filtered_scores else 0

            return jsonify({
                "results": results,
                "threshold": threshold,
                "submission": submission_dict,
                "exam": exam_dict,
                "overall_score": overall_score
            })


        submission_id = request.args.get('submission') or current_user.id
        return self.view("result", submission_id=submission_id, required_roles="all")
