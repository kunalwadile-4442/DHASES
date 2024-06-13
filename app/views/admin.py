from flask import g
from flask_login import current_user, login_required
from sqlalchemy import MetaData, text

from app.custom import CustomBlueprint
from app.models import db, Exam, User, Invitation

class AdminView(CustomBlueprint):
    def __init__(self):
        super().__init__("admin", __name__, url_prefix="/admin")

        self.before_request(self.before_request_hook) 
        self.route("/")(self.index)

    def index(self):
        all_exams = Exam.query.all()
        all_users = User.query.all()
        all_invitations = Invitation.query.all()

        stats = {}

        def is_active(exam):
            return exam.is_finished == False
        
        def is_finished(exam):
            return exam.is_finished == True
        
        def is_teacher(user):
            return user.role == "teacher"
        
        def is_student(user):
            return user.role == "student"
        
        def is_admin(user):
            return user.role == "admin"

        def invited_by_me(invitation):
            return invitation.invited_by == current_user.id
        
        def is_used(invitation):
            return invitation.is_used == True
        
        stats["total_exams"] = len(all_exams)
        stats["active_exams"] = len(list(filter(is_active, all_exams)))
        stats["finished_exams"] = len(list(filter(is_finished, all_exams)))
        stats["total_users"] = len(all_users)
        stats["total_teachers"] = len(list(filter(is_teacher, all_users)))
        stats["total_students"] = len(list(filter(is_student, all_users)))
        stats["total_admins"] = len(list(filter(is_admin, all_users)))
        stats["total_invitations"] = len(all_invitations)
        stats["invitations_sent_by_me"] = len(list(filter(invited_by_me, all_invitations)))
        stats["invitations_used"] = len(list(filter(is_used, all_invitations)))

        return self.view("overview", stats=stats)
    
    @login_required
    def before_request_hook(self):

        # Reflect the database tables using MetaData
        meta = MetaData()
        meta.reflect(bind=db.engine)

        # Get a list of all table names
        g.table_names = meta.tables.keys()
