"""
project/app.py
--------------
Main entry point for the Flask-based ASD Meltdown Predictor application.
"""

import os
from flask import Flask
from routes import bp

def create_app() -> Flask:
    """Application factory for configuring and instantiating the Flask app."""
    app = Flask(__name__)

    # Set application secret key for sessions
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "prohealth_clinical_secret_key_2026")

    # Register the main route blueprints
    app.register_blueprint(bp)

    return app

app = create_app()

if __name__ == "__main__":
    # Start database connection check on boot
    from database import test_connection
    db_status = test_connection()
    if db_status["success"]:
        print(f"[DATABASE] Connection established successfully: {db_status['database']} on {db_status['host']}")
    else:
        print(f"[DATABASE] Connection check warning: {db_status['message']}")
        print("   Make sure your database server is running and configurations are loaded correctly.")

    # Start Flask development server
    port = int(os.getenv("PORT", 5000))
    print(f"[SERVER] Starting Flask on port {port}... Debug mode: True (Reloader: Off)")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
