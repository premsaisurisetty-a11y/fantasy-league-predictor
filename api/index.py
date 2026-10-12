import os
import sys
import urllib.parse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.app import app
from flask import request, jsonify

@app.before_request
def debug_capture():
    if "debug" in request.args:
        return jsonify({
            "path": request.path,
            "query_string": request.environ.get("QUERY_STRING"),
            "path_info": request.environ.get("PATH_INFO"),
            "args": dict(request.args),
            "headers": {k: v for k, v in request.headers.items() if not k.lower().startswith("x-vercel-oidc")}
        })

class VercelPathFix:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        qs = environ.get("QUERY_STRING", "")
        if "__orig_path=" in qs:
            params = urllib.parse.parse_qs(qs)
            if "__orig_path" in params and params["__orig_path"]:
                target_path = params["__orig_path"][0]
                if target_path.startswith("//"):
                    target_path = target_path[1:]
                environ["PATH_INFO"] = target_path
        elif environ.get("PATH_INFO") in ["/api/index.py", "/api/index"]:
            environ["PATH_INFO"] = "/"
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFix(app.wsgi_app)
