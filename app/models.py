from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import random
import pytz

db = SQLAlchemy()
timezone = pytz.timezone("Asia/Kolkata")

USER_ROLES = ("admin", "teacher", "student")


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.Enum(*USER_ROLES), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now(timezone))
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Exam(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    answers = db.Column(db.Text, nullable=False)  # JSON or another suitable format
    file_path = db.Column(db.String(100), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    teacher = db.relationship("User", backref="exams", lazy=True)
    total_marks = db.Column(db.Float, nullable=False)
    threshold = db.Column(db.Float, nullable=False, default=0.7)
    is_finished = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now(timezone))

class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    student = db.relationship("User", backref="submissions", lazy=True)
    exam_id = db.Column(db.Integer, db.ForeignKey("exam.id"), nullable=False)
    exam = db.relationship("Exam", backref="submissions", lazy=True)
    file_path = db.Column(db.String(100), nullable=False)
    submitted_answers = db.Column(
        db.Text, nullable=False
    )  # JSON or another suitable format
    similarity_scores = db.Column(
        db.Text, nullable=False
    )  # JSON or another suitable format
    manual_marks = db.Column(db.Float)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now(timezone))

class Invitation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    role = db.Column(db.Enum("teacher", "student"), nullable=False)
    code = db.Column(db.String(50), nullable=False)
    is_used = db.Column(db.Boolean, nullable=False, default=False)
    invited_by = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now(timezone))
    def __init__(self, email, role, invited_by):
        self.email = email
        self.role = role
        self.invited_by = invited_by
        self.code = random.randint(100000, 999999)
        self.is_used = False
    
class User_kunal(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.Enum(*USER_ROLES), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now(timezone))
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)