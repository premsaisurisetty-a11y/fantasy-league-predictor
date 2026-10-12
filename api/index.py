import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from app.app import app
except Exception as e:
    import traceback
    from flask import Flask, jsonify
    err_msg = str(e)
    err_tb = traceback.format_exc()
    app = Flask(__name__)
    
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def catch_all(path):
        return jsonify({
            "status": "error",
            "message": err_msg,
            "traceback": err_tb
        }), 500
