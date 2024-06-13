import logging

from flask_login import LoginManager
from flask_caching import Cache

# Login Manager
lm = LoginManager()

# cache
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})

# Logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Helper functions

def get_time_of_day():
    """Returns the time of day as a string."""
    from datetime import datetime

    current_hour = datetime.now().hour

    if current_hour < 12:
        return "morning"
    elif current_hour < 18:
        return "afternoon"
    else:
        return "evening"
    
def get_user_role(user):
    """Returns the role of a user as a string."""
    if user.is_authenticated:
        return user.role
    else:
        return "anonymous"