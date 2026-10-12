import os
import sys
import traceback

# Ensure the root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from app.app import app
    handler = app
except Exception as e:
    err_traceback = traceback.format_exc()
    print(f"[Vercel Init Error] {err_traceback}", file=sys.stderr)
    
    from flask import Flask, Response
    fallback_app = Flask(__name__)
    
    @fallback_app.route("/", defaults={"path": ""})
    @fallback_app.route("/<path:path>")
    def catch_all(path):
        content = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Startup Error - FantasyScore AI</title></head>
        <body style="background:#0f172a; color:#f8fafc; font-family:monospace; padding:24px;">
            <h2 style="color:#ef4444;">Serverless Initialization Exception</h2>
            <p>The Python application failed to initialize. Traceback:</p>
            <pre style="background:#1e293b; color:#38bdf8; padding:16px; border-radius:8px; overflow:auto;">{err_traceback}</pre>
            <hr style="border-color:#334155; margin:20px 0;">
            <p style="color:#94a3b8;">BASE_DIR: {BASE_DIR}</p>
            <p style="color:#94a3b8;">Files in BASE_DIR: {os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else 'Not Found'}</p>
        </body>
        </html>
        """
        return Response(content, status=500, mimetype="text/html")
        
    handler = fallback_app
