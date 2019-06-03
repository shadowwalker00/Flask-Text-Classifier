import os
CSRF_ENABLED = True
SECRET_KEY = 'you-will-never-guess'

SECRET_KEY = os.urandom(24)