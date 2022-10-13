# -*- coding: utf-8 -*-
"""This module creates a wsgi interface to be used with gunicorn"""

from flask_server.app import app

if __name__ == "__main__":
    app.run()
