import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.app import app
from flask import request, jsonify

@app.route("/ping")
def ping():
    return "pong"

@app.route("/debug")
def debug_route():
    return jsonify({
        "path": request.path,
        "environ_path_info": request.environ.get("PATH_INFO"),
        "headers": dict(request.headers),
        "rules": [str(r) for r in app.url_map.iter_rules()]
    })

class VercelPathFix:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        for prefix in ["/api/index.py", "/api/index"]:
            if path == prefix:
                path = "/"
                break
            elif path.startswith(prefix + "/"):
                path = path[len(prefix):]
                break
        environ["PATH_INFO"] = path
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFix(app.wsgi_app)
