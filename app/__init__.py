from flask import Flask
from flask_migrate import Migrate

from .models import User, db
from .utils import lm, cache
from .views import AdminView, CommonView, StudentView, TeacherView

app = Flask(__name__)
app.config["SECRET_KEY"] = "your_secret_key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///evalsheet.db"

db.init_app(app)
lm.init_app(app)
cache.init_app(app)
lm.login_view = "common.login"
migrate = Migrate(app, db)

@lm.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


app.register_blueprint(AdminView())
app.register_blueprint(CommonView())
app.register_blueprint(TeacherView())
app.register_blueprint(StudentView())
