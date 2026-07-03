"""
project/routes.py
-----------------
Defines the Flask application routes, session handlers, database lookups, 
and machine learning scoring controllers.
"""

from flask import Blueprint, render_template, request, redirect, url_for, session, Response, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from predictor import predict
from suggestions import get_suggestions
from models import (
    create_user, get_user_by_email, email_exists,
    create_child_profile, get_child_profiles, get_child_profile_by_id,
    save_prediction, get_prediction_history, get_prediction_by_id,
    get_child_weekly_trends, update_prediction_notes
)

bp = Blueprint("main", __name__)

@bp.route("/")
def landing():
    """Renders the public landing / introduction page."""
    if session.get("user"):
        return redirect(url_for("main.dashboard"))
    return render_template("landing.html")

@bp.route("/login", methods=["GET", "POST"])
def login():
    """Handles caregiver user login requests via database credentials."""
    if session.get("user"):
        return redirect(url_for("main.dashboard"))
        
    error = None
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        
        user = get_user_by_email(email)
        if user and check_password_hash(user["password"], password):
            session["user"] = {
                "id": user["id"],
                "name": user["full_name"],
                "email": user["email"],
                "role": "Parent/Caregiver"
            }
            return redirect(url_for("main.dashboard"))
        else:
            error = "Invalid email address or password."
            
    return render_template("login.html", error=error)

@bp.route("/register", methods=["GET", "POST"])
def register():
    """Handles caregiver user registration requests using Werkzeug hash encryption."""
    if session.get("user"):
        return redirect(url_for("main.dashboard"))
        
    error = None
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        
        if email_exists(email):
            error = "Email address already registered. Please sign in."
        else:
            hashed_password = generate_password_hash(password)
            user_id = create_user(name, email, hashed_password)
            if user_id != -1:
                # Instantly login the newly registered user
                session["user"] = {
                    "id": user_id,
                    "name": name,
                    "email": email,
                    "role": "Parent/Caregiver"
                }
                return redirect(url_for("main.dashboard"))
            else:
                error = "An error occurred during registration. Please try again."
                
    return render_template("register.html", error=error)

@bp.route("/logout")
def logout():
    """Clears user session variables and redirects to landing page."""
    session.clear()
    return redirect(url_for("main.landing"))

@bp.route("/dashboard")
def dashboard():
    """Renders the caregiver dashboard, fetching registered child profiles."""
    if not session.get("user"):
        return redirect(url_for("main.login"))
        
    user_id = session["user"]["id"]
    profiles = get_child_profiles(user_id)
    return render_template("dashboard.html", profiles=profiles)

@bp.route("/child/add", methods=["POST"])
def add_child():
    """Handles child profile creation requests linked to the caregiver."""
    if not session.get("user"):
        return redirect(url_for("main.login"))
        
    user_id = session["user"]["id"]
    child_name = request.form.get("child_name")
    age = int(request.form.get("child_age", 8))
    gender = request.form.get("gender")
    notes = request.form.get("notes", "")
    address = request.form.get("address", "")
    
    create_child_profile(user_id, child_name, age, gender, notes, address)
    return redirect(url_for("main.profile"))

@bp.route("/assessment", methods=["GET", "POST"])
def assessment():
    """Renders form fields and runs ML prediction models on submission."""
    if not session.get("user"):
        return redirect(url_for("main.login"))
        
    user_id = session["user"]["id"]
    profiles = get_child_profiles(user_id)
    
    results = None
    inputs = None
    selected_child_id = None
    
    if request.method == "POST":
        try:
            selected_child_id = int(request.form.get("child_id"))
            
            # Fetch child profile directly from database to get registered age
            child = get_child_profile_by_id(selected_child_id, user_id)
            if not child:
                return redirect(url_for("main.dashboard"))
                
            inputs = {
                "age":                int(child["age"]), # Retrieve age from child profile
                "sleep_hours":        float(request.form.get("sleep_hours", 7.0)),
                "stress_level":       int(request.form.get("stress_level", 5)),
                "social_interaction": int(request.form.get("social_interaction", 5)),
                "sensory_sensitivity":int(request.form.get("sensory_sensitivity", 5)),
                "anxiety_level":      int(request.form.get("anxiety_level", 5)),
                "routine_change":     int(request.form.get("routine_change", 0)),
                "noise_tolerance":    int(request.form.get("noise_tolerance", 5)),
            }
            
            # Execute predictions using ML interface
            probability, label = predict(inputs)
            suggestions_data = get_suggestions(probability)
            
            # Save results into predictions table in MySQL
            new_pred_id = save_prediction(
                user_id=user_id,
                child_id=selected_child_id,
                sleep_hours=inputs["sleep_hours"],
                stress_level=inputs["stress_level"],
                social_interaction=inputs["social_interaction"],
                sensory_sensitivity=inputs["sensory_sensitivity"],
                anxiety_level=inputs["anxiety_level"],
                routine_change=(inputs["routine_change"] == 1),
                noise_tolerance=inputs["noise_tolerance"],
                probability=probability,
                risk_level=suggestions_data["risk_level"],
                prediction_result="Meltdown Likely" if label == 1 else "No Meltdown"
            )
            
            results = {
                "pred_id":        new_pred_id,
                "probability":    probability,
                "predicted":      label,
                "risk_level":     suggestions_data["risk_level"],
                "overall":        suggestions_data["overall"],
                "suggestions":    suggestions_data["suggestions"],
                "classification": "Likely Meltdown" if label == 1 else "Unlikely Meltdown"
            }
        except Exception as e:
            print(f"[ERROR] Live database prediction failed: {e}")
            
    return render_template("assessment.html", profiles=profiles, results=results, inputs=inputs, selected_child_id=selected_child_id)

@bp.route("/history")
def history():
    """Fetches prediction logs history records from database."""
    if not session.get("user"):
        return redirect(url_for("main.login"))
        
    user_id = session["user"]["id"]
    history_logs = get_prediction_history(user_id)
    return render_template("history.html", history=history_logs)

@bp.route("/download/<int:pred_id>")
def download_report(pred_id):
    """Compiles prediction metrics and suggestions into a caregiver report file."""
    if not session.get("user"):
        return redirect(url_for("main.login"))
        
    user_id = session["user"]["id"]
    record = get_prediction_by_id(pred_id, user_id)
    if not record:
        return redirect(url_for("main.history"))
        
    # Compile assessment report text content
    date_str = record['created_at'].strftime('%Y-%m-%d %H:%M') if record['created_at'] else 'N/A'
    gender_str = record['gender'] if record['gender'] else 'N/A'
    notes_str = record['child_notes'] if record['child_notes'] else 'None'
    routine_str = "Disrupted / Routine Changed" if record['routine_change'] else "Stable / No Disruption"
    
    report = f"""==================================================
           PROHEALTH ASD RISK ASSESSMENT REPORT
==================================================
Date generated : {date_str}
Caregiver/Parent : {session['user']['name']}
Child Name     : {record['child_name']}
Child Age      : {record['age']} years
Child Gender   : {gender_str}
Child Notes    : {notes_str}
--------------------------------------------------
Calculated Probability : {record['probability']}%
Calculated Risk Level  : {record['risk_level'].replace("🟢 ", "").replace("🟡 ", "").replace("🔴 ", "")}
Classification Output  : {record['prediction_result']}
--------------------------------------------------
Assessed Environmental & Behavioral Factors:
  - Sleep Duration       : {record['sleep_hours']} hours
  - Stress Level Today   : {record['stress_level']}/10
  - Anxiety Level Today  : {record['anxiety_level']}/10
  - Social Interaction   : {record['social_interaction']}/10
  - Sensory Sensitivity  : {record['sensory_sensitivity']}/10
  - Noise Tolerance      : {record['noise_tolerance']}/10
  - Daily Routine Change : {routine_str}
--------------------------------------------------
Recommended Suggestions:
"""
    # Fetch suggestions dynamically
    sugg_res = get_suggestions(record['probability'])
    for topic, text in sugg_res["suggestions"].items():
        report += f"\n* [{topic}]\n  {text}\n"
        
    report += "\n==================================================\n"
    
    # Format Response headers for file attachment
    safe_name = record['child_name'].replace(" ", "_")
    filename = f"prohealth_report_{safe_name}_{record['created_at'].strftime('%Y%m%d')}.txt"
    
    return Response(
        report,
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )

@bp.route("/profile", methods=["GET", "POST"])
def profile():
    """Renders the caregiver user profile details."""
    if not session.get("user"):
        return redirect(url_for("main.login"))
    user_id = session["user"]["id"]
    profiles = get_child_profiles(user_id)
    return render_template("profile.html", profiles=profiles)

@bp.route("/prediction/notes/<int:pred_id>", methods=["POST"])
def save_notes(pred_id):
    """Saves or updates caregiver notes for a specific prediction assessment day."""
    if not session.get("user"):
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    user_id = session["user"]["id"]
    notes = request.form.get("notes", "")
    success = update_prediction_notes(pred_id, user_id, notes)
    if success:
        return jsonify({"success": True, "message": "Notes saved successfully."})
    return jsonify({"success": False, "message": "Failed to save notes."}), 500

@bp.route("/api/trends/<int:child_id>")
def child_trends(child_id):
    """Returns the last 7 predictions data for the specified child to feed Chart.js."""
    if not session.get("user"):
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    user_id = session["user"]["id"]
    
    # Verify child ownership
    child = get_child_profile_by_id(child_id, user_id)
    if not child:
        return jsonify({"success": False, "message": "Child profile not found."}), 404
        
    trends = get_child_weekly_trends(child_id, user_id)
    
    data = []
    for t in trends:
        date_str = t["created_at"].strftime("%m-%d") if t["created_at"] else "N/A"
        data.append({
            "date": date_str,
            "probability": t["probability"],
            "risk_level": t["risk_level"].replace("🟢 ", "").replace("🟡 ", "").replace("🔴 ", "")
        })
        
    return jsonify({"success": True, "trends": data})
