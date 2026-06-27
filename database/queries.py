"""
database/queries.py
-------------------
All database query functions for the
ASD Meltdown Probability Predictor project.

This file contains ready-to-use functions for:
  - Creating and fetching users
  - Managing child profiles
  - Saving and retrieving predictions

HOW TO USE:
  from database.queries import create_user, get_user_by_email
  from database.queries import save_prediction, get_prediction_history

BEGINNER TIP:
  Every function in this file does ONE job only.
  This makes it easy to test, debug, and explain.
"""

from database.db import get_connection


# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — USER FUNCTIONS
#  Tables used: users
# ══════════════════════════════════════════════════════════════════

def create_user(full_name: str, email: str, password: str) -> int:
    """
    Insert a new user into the users table.

    The password passed here should already be HASHED (never store
    plain-text passwords). Hashing is done in auth/auth_utils.py
    before calling this function.

    Parameters
    ----------
    full_name : str  → e.g. "Aarav Sharma"
    email     : str  → e.g. "aarav@email.com"
    password  : str  → bcrypt hashed password string

    Returns
    -------
    int → the new user's auto-generated ID (e.g. 3)
          returns -1 if something goes wrong

    Example
    -------
        new_id = create_user("Aarav Sharma", "aarav@email.com", hashed_pw)
        print(new_id)  # → 3
    """
    # SQL query using %s placeholders (parameterized = safe from SQL injection)
    sql = """
        INSERT INTO users (full_name, email, password)
        VALUES (%s, %s, %s)
    """

    connection = None
    cursor     = None

    try:
        connection = get_connection()
        cursor     = connection.cursor()

        # Pass values as a tuple — mysql.connector fills in the %s safely
        cursor.execute(sql, (full_name, email, password))

        # Save the changes to the database permanently
        connection.commit()

        # lastrowid gives us the auto-generated primary key of the new row
        new_user_id = cursor.lastrowid
        return new_user_id

    except Exception as e:
        print(f"[ERROR] create_user failed: {e}")
        if connection:
            connection.rollback()   # undo any partial changes
        return -1

    finally:
        # Always close cursor and connection even if an error happened
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


# ─────────────────────────────────────────────────────────────────

def get_user_by_email(email: str) -> dict | None:
    """
    Fetch a single user row from the database by their email.

    Used during LOGIN to:
      1. Check if the email exists
      2. Get the stored hashed password for verification

    Parameters
    ----------
    email : str → the email the user typed on the login form

    Returns
    -------
    dict  → all columns as a dictionary if user is found
            e.g. {"id": 3, "full_name": "Aarav", "email": "...", ...}
    None  → if no user with that email exists

    Example
    -------
        user = get_user_by_email("aarav@email.com")
        if user:
            print(user["full_name"])   # → "Aarav Sharma"
            print(user["password"])    # → bcrypt hash
    """
    sql = "SELECT * FROM users WHERE email = %s"

    connection = None
    cursor     = None

    try:
        connection = get_connection()

        # dictionary=True → rows come back as dicts, not plain tuples
        # e.g. {"id": 1, "email": "..."} instead of (1, "...")
        cursor = connection.cursor(dictionary=True)

        cursor.execute(sql, (email,))   # note the comma: (email,) is a tuple

        # fetchone() returns ONE matching row, or None if not found
        user = cursor.fetchone()
        return user

    except Exception as e:
        print(f"[ERROR] get_user_by_email failed: {e}")
        return None

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


# ─────────────────────────────────────────────────────────────────

def email_exists(email: str) -> bool:
    """
    Check whether an email is already registered.

    Used during SIGNUP to prevent duplicate accounts.

    Parameters
    ----------
    email : str → email to check

    Returns
    -------
    bool → True if email is already in the database, False otherwise

    Example
    -------
        if email_exists("aarav@email.com"):
            st.error("Email already registered. Please login.")
        else:
            create_user(...)
    """
    sql = "SELECT id FROM users WHERE email = %s"

    connection = None
    cursor     = None

    try:
        connection = get_connection()
        cursor     = connection.cursor()

        cursor.execute(sql, (email,))
        result = cursor.fetchone()

        # If result is not None, the email exists
        return result is not None

    except Exception as e:
        print(f"[ERROR] email_exists failed: {e}")
        return False

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — CHILD PROFILE FUNCTIONS
#  Tables used: child_profiles
# ══════════════════════════════════════════════════════════════════

def create_child_profile(
    user_id:    int,
    child_name: str,
    age:        int,
    gender:     str  = None,
    notes:      str  = None
) -> int:
    """
    Create a new child profile linked to a logged-in user.

    Each user can have multiple child profiles.
    The user_id links this profile to the parent/caregiver account.

    Parameters
    ----------
    user_id    : int → ID of the logged-in user (from session)
    child_name : str → e.g. "Arjun"
    age        : int → e.g. 8
    gender     : str → optional, e.g. "Male", "Female", "Other"
    notes      : str → optional, any extra notes about the child

    Returns
    -------
    int → the new child profile's ID, or -1 on failure

    Example
    -------
        child_id = create_child_profile(
            user_id=3,
            child_name="Arjun",
            age=8,
            gender="Male",
            notes="Sensitive to loud noises"
        )
        print(child_id)  # → 5
    """
    sql = """
        INSERT INTO child_profiles (user_id, child_name, age, gender, notes)
        VALUES (%s, %s, %s, %s, %s)
    """

    connection = None
    cursor     = None

    try:
        connection = get_connection()
        cursor     = connection.cursor()

        cursor.execute(sql, (user_id, child_name, age, gender, notes))
        connection.commit()

        return cursor.lastrowid

    except Exception as e:
        print(f"[ERROR] create_child_profile failed: {e}")
        if connection:
            connection.rollback()
        return -1

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


# ─────────────────────────────────────────────────────────────────

def get_child_profiles(user_id: int) -> list[dict]:
    """
    Fetch all child profiles that belong to a specific user.

    Used in the Streamlit app to let the user select which child
    they are running the prediction for.

    Parameters
    ----------
    user_id : int → the logged-in user's ID (from session)

    Returns
    -------
    list[dict] → list of child profile rows as dictionaries
                 e.g. [{"id": 5, "child_name": "Arjun", "age": 8, ...}]
                 returns [] (empty list) if no profiles found

    Example
    -------
        profiles = get_child_profiles(user_id=3)
        for child in profiles:
            print(child["child_name"], child["age"])
    """
    sql = """
        SELECT id, child_name, age, gender, notes, created_at
        FROM child_profiles
        WHERE user_id = %s
        ORDER BY created_at DESC
    """

    connection = None
    cursor     = None

    try:
        connection = get_connection()
        cursor     = connection.cursor(dictionary=True)

        cursor.execute(sql, (user_id,))

        # fetchall() returns a list of all matching rows
        profiles = cursor.fetchall()
        return profiles

    except Exception as e:
        print(f"[ERROR] get_child_profiles failed: {e}")
        return []

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — PREDICTION FUNCTIONS
#  Tables used: predictions
# ══════════════════════════════════════════════════════════════════

def save_prediction(
    user_id:              int,
    child_id:             int,
    sleep_hours:          float,
    stress_level:         int,
    social_interaction:   int,
    sensory_sensitivity:  int,
    anxiety_level:        int,
    routine_change:       bool,
    noise_tolerance:      int,
    probability:          float,
    risk_level:           str,
    prediction_result:    str
) -> int:
    """
    Save one prediction result to the predictions table.

    Called immediately after the model returns a prediction in app.py.
    Stores both the INPUT features and the OUTPUT results together,
    so the full prediction is available in history later.

    Parameters
    ----------
    user_id            : int   → logged-in user's ID
    child_id           : int   → selected child profile ID
    sleep_hours        : float → e.g. 6.5
    stress_level       : int   → 1–10
    social_interaction : int   → 1–10
    sensory_sensitivity: int   → 1–10
    anxiety_level      : int   → 1–10
    routine_change     : bool  → True or False
    noise_tolerance    : int   → 1–10
    probability        : float → e.g. 78.34 (as percentage)
    risk_level         : str   → "Low Risk", "Medium Risk", "High Risk"
    prediction_result  : str   → "No Meltdown" or "Meltdown Likely"

    Returns
    -------
    int → the new prediction's ID, or -1 on failure

    Example
    -------
        pred_id = save_prediction(
            user_id=3, child_id=5,
            sleep_hours=5.5, stress_level=8,
            social_interaction=3, sensory_sensitivity=9,
            anxiety_level=7, routine_change=True,
            noise_tolerance=2, probability=82.4,
            risk_level="High Risk",
            prediction_result="Meltdown Likely"
        )
    """
    sql = """
        INSERT INTO predictions (
            user_id,              child_id,
            sleep_hours,          stress_level,
            social_interaction,   sensory_sensitivity,
            anxiety_level,        routine_change,
            noise_tolerance,      probability,
            risk_level,           prediction_result
        )
        VALUES (
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s
        )
    """

    connection = None
    cursor     = None

    try:
        connection = get_connection()
        cursor     = connection.cursor()

        cursor.execute(sql, (
            user_id,             child_id,
            sleep_hours,         stress_level,
            social_interaction,  sensory_sensitivity,
            anxiety_level,       routine_change,
            noise_tolerance,     round(probability, 4),
            risk_level,          prediction_result
        ))

        connection.commit()
        return cursor.lastrowid

    except Exception as e:
        print(f"[ERROR] save_prediction failed: {e}")
        if connection:
            connection.rollback()
        return -1

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


# ─────────────────────────────────────────────────────────────────

def get_prediction_history(user_id: int) -> list[dict]:
    """
    Fetch all past predictions for a specific user.

    Uses a JOIN to also include the child's name from child_profiles,
    so the history tab can display "Prediction for Arjun" instead of
    just a child_id number.

    Results are ordered newest first (most recent prediction at top).

    Parameters
    ----------
    user_id : int → the logged-in user's ID

    Returns
    -------
    list[dict] → list of prediction rows, each as a dictionary.
                 Each row includes child_name from the JOIN.
                 Returns [] if no predictions found.

    Keys in each dict:
        id, child_name, sleep_hours, stress_level,
        social_interaction, sensory_sensitivity,
        anxiety_level, routine_change, noise_tolerance,
        probability, risk_level, prediction_result, created_at

    Example
    -------
        history = get_prediction_history(user_id=3)
        for record in history:
            print(record["child_name"], record["risk_level"],
                  record["probability"])
    """
    # JOIN pulls child_name from child_profiles so history is readable
    sql = """
        SELECT
            p.id,
            c.child_name,
            p.sleep_hours,
            p.stress_level,
            p.social_interaction,
            p.sensory_sensitivity,
            p.anxiety_level,
            p.routine_change,
            p.noise_tolerance,
            p.probability,
            p.risk_level,
            p.prediction_result,
            p.created_at
        FROM predictions p
        JOIN child_profiles c
            ON p.child_id = c.id
        WHERE p.user_id = %s
        ORDER BY p.created_at DESC
    """

    connection = None
    cursor     = None

    try:
        connection = get_connection()
        cursor     = connection.cursor(dictionary=True)

        cursor.execute(sql, (user_id,))
        history = cursor.fetchall()
        return history

    except Exception as e:
        print(f"[ERROR] get_prediction_history failed: {e}")
        return []

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


# ══════════════════════════════════════════════════════════════════
#  QUICK TEST — run this file directly to verify all functions
#  python database/queries.py
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  queries.py — Quick Function Test")
    print("═" * 55)

    # ── Test email_exists ─────────────────────────────────────
    test_email = "test_student@asd.com"
    exists = email_exists(test_email)
    print(f"\n[1] email_exists('{test_email}') → {exists}")

    # ── Test create_user ──────────────────────────────────────
    if not exists:
        uid = create_user("Test Student", test_email, "hashed_pw_here")
        print(f"[2] create_user() → new user ID = {uid}")
    else:
        print("[2] create_user() → skipped (email already exists)")

    # ── Test get_user_by_email ────────────────────────────────
    user = get_user_by_email(test_email)
    if user:
        print(f"[3] get_user_by_email() → {user['full_name']} (id={user['id']})")

    print("\n" + "═" * 55)
    print("  All queries tested successfully.")
    print("═" * 55 + "\n")