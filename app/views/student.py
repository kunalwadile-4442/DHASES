import os

from flask import g
from flask_login import current_user, login_required

from app.custom import CustomBlueprint
from app.utils import get_time_of_day
from app.models import Exam, Submission

import json
from datetime import datetime

class StudentView(CustomBlueprint):
    def __init__(self):
        super().__init__("student", __name__, url_prefix="/student")

        self.upload_folder = os.path.join(os.path.dirname(__file__), 'uploads')

        self.before_request(self.before_request_hook) 
        self.route("/")(self.index)

    def index(self):
        all_exams = Exam.query.all()
        all_submissions = Submission.query.filter_by(student_id=current_user.id).all()
        unsubmitted_exams = list(filter(lambda exam: exam.id not in [submission.exam_id for submission in all_submissions], all_exams))

        stats = {}

        def is_active(exam):
            return exam.is_finished == False
        
        def is_finished(exam):
            return exam.is_finished == True
        
        stats["total_exams"] = len(all_exams)
        stats["active_exams"] = len(list(filter(is_active, all_exams)))
        stats["finished_exams"] = len(list(filter(is_finished, all_exams)))
        stats["total_submissions"] = len(all_submissions)
        stats["pending_submissions"] = len(list(filter(is_active, unsubmitted_exams)))
        stats["missed_submissions"] = len(list(filter(is_finished, unsubmitted_exams)))
    
        return self.view("overview", stats=stats)
    
    @login_required
    def before_request_hook(self):
        g.greetings = "Good {time_of_day}, {name}!".format(
            time_of_day=get_time_of_day(),
            name=current_user.username,
        )