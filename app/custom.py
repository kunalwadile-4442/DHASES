from flask import Blueprint, render_template, redirect, url_for
from flask_login import current_user
from app.models import USER_ROLES

class CustomBlueprint(Blueprint):
    def __init__(self, name, import_name, url_prefix):
        super().__init__(name, import_name, url_prefix=url_prefix)

    def view(self, template, required_roles='default', **kwargs):
        if required_roles == 'default':
            required_roles = (self.name,) if (current_user and current_user.is_authenticated) else None
        elif required_roles == 'all':
            required_roles = USER_ROLES

        if required_roles and current_user and current_user.role not in required_roles:
            print(f"User {current_user} tried to access {template} without permission")
            return redirect(url_for(f"{current_user.role}.index"))
            
        template_path = f"{self.name.lower()}/{template}.html"
        return render_template(template_path, **kwargs)
