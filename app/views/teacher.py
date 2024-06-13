from flask import g
from flask_login import current_user, login_required

from app.custom import CustomBlueprint
from app.utils import get_time_of_day
from app.models import Exam, User, Invitation

class TeacherView(CustomBlueprint):
    def __init__(self):
        super().__init__("teacher", __name__, url_prefix="/teacher")

        self.before_request(self.before_request_hook) 
        self.route("/")(self.index)

    def index(self):
        all_exams = Exam.query.filter_by(teacher_id=current_user.id).all()
        all_students = User.query.filter_by(role="student").all()
        all_invitations = Invitation.query.filter_by(invited_by=current_user.id).all()

        stats = {}

        def is_active(exam):
            return exam.is_finished == False
        
        def is_finished(exam):
            return exam.is_finished == True
        
        stats["total_exams"] = len(all_exams)
        stats["active_exams"] = len(list(filter(is_active, all_exams)))
        stats["finished_exams"] = len(list(filter(is_finished, all_exams)))
        stats["total_students"] = len(all_students)
        stats["invitations_sent"] = len(all_invitations)
        stats["students_joined"] = len(list(filter(lambda invitation: invitation.is_used == True, all_invitations)))

        return self.view("overview", stats=stats)
    
    @login_required
    def before_request_hook(self):
        g.greetings = "Good {time_of_day}, {name}!".format(
            time_of_day=get_time_of_day(),
            name=current_user.username,
        )