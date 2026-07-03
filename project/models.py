"""
project/models.py
-----------------
Defines all SQL interaction queries for users, child profiles, and predictions.
"""

from database import get_connection

# ══════════════════════════════════════════════════════════════════
# USER MANAGEMENT FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def create_user(full_name: str, email: str, password: str) -> int:
    """Insert a new user into the database. Password should already be hashed."""
    sql = """
        INSERT INTO users (full_name, email, password)
        VALUES (%s, %s, %s)
    """
    connection = None
    cursor     = None
    try:
        connection = get_connection()
        cursor     = connection.cursor()
        cursor.execute(sql, (full_name, email, password))
        connection.commit()
        return cursor.lastrowid
    except Exception as e:
        print(f"[ERROR] create_user failed: {e}")
        if connection:
            connection.rollback()
        return -1
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

def get_user_by_email(email: str) -> dict | None:
    """Fetch a user record by email address."""
    sql = "SELECT * FROM users WHERE email = %s"
    connection = None
    cursor     = None
    try:
        connection = get_connection()
        cursor     = connection.cursor(dictionary=True)
        cursor.execute(sql, (email,))
        return cursor.fetchone()
    except Exception as e:
        print(f"[ERROR] get_user_by_email failed: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

def email_exists(email: str) -> bool:
    """Check if an email has already been registered."""
    sql = "SELECT id FROM users WHERE email = %s"
    connection = None
    cursor     = None
    try:
        connection = get_connection()
        cursor     = connection.cursor()
        cursor.execute(sql, (email,))
        return cursor.fetchone() is not None
    except Exception as e:
        print(f"[ERROR] email_exists failed: {e}")
        return False
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

# ══════════════════════════════════════════════════════════════════
# CHILD PROFILE FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def create_child_profile(
    user_id:    int,
    child_name: str,
    age:        int,
    gender:     str  = None,
    notes:      str  = None,
    address:    str  = None
) -> int:
    """Create a child profile linked to the parent account."""
    sql = """
        INSERT INTO child_profiles (user_id, child_name, age, gender, notes, address)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    connection = None
    cursor     = None
    try:
        connection = get_connection()
        cursor     = connection.cursor()
        cursor.execute(sql, (user_id, child_name, age, gender, notes, address))
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

def get_child_profiles(user_id: int) -> list[dict]:
    """Retrieve all child profiles belonging to a caregiver user."""
    sql = """
        SELECT id, child_name, age, gender, notes, address, created_at
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
        return cursor.fetchall()
    except Exception as e:
        print(f"[ERROR] get_child_profiles failed: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

# ══════════════════════════════════════════════════════════════════
# PREDICTION FUNCTIONS
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
    """Save a run prediction score and its associated inputs."""
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
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
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

def get_prediction_history(user_id: int) -> list[dict]:
    """Retrieve user's historical predictions joined with child names."""
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
            p.notes,
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
        return cursor.fetchall()
    except Exception as e:
        print(f"[ERROR] get_prediction_history failed: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

def get_child_profile_by_id(child_id: int, user_id: int) -> dict | None:
    """Fetch a single child profile by ID, verifying caregiver ownership."""
    sql = """
        SELECT id, child_name, age, gender, notes, address
        FROM child_profiles
        WHERE id = %s AND user_id = %s
    """
    connection = None
    cursor     = None
    try:
        connection = get_connection()
        cursor     = connection.cursor(dictionary=True)
        cursor.execute(sql, (child_id, user_id))
        return cursor.fetchone()
    except Exception as e:
        print(f"[ERROR] get_child_profile_by_id failed: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

def get_prediction_by_id(pred_id: int, user_id: int) -> dict | None:
    """Fetch a single prediction record, verifying caregiver ownership, joined with child metadata."""
    sql = """
        SELECT
            p.id,
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
            p.notes,
            p.created_at,
            c.child_name,
            c.age,
            c.gender,
            c.notes AS child_notes
        FROM predictions p
        JOIN child_profiles c
            ON p.child_id = c.id
        WHERE p.id = %s AND p.user_id = %s
    """
    connection = None
    cursor     = None
    try:
        connection = get_connection()
        cursor     = connection.cursor(dictionary=True)
        cursor.execute(sql, (pred_id, user_id))
        return cursor.fetchone()
    except Exception as e:
        print(f"[ERROR] get_prediction_by_id failed: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

def update_prediction_notes(pred_id: int, user_id: int, notes: str) -> bool:
    """Update caregiver daily notes for a prediction record."""
    sql = """
        UPDATE predictions
        SET notes = %s
        WHERE id = %s AND user_id = %s
    """
    connection = None
    cursor     = None
    try:
        connection = get_connection()
        cursor     = connection.cursor()
        cursor.execute(sql, (notes, pred_id, user_id))
        connection.commit()
        return True
    except Exception as e:
        print(f"[ERROR] update_prediction_notes failed: {e}")
        if connection:
            connection.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

def get_child_weekly_trends(child_id: int, user_id: int) -> list[dict]:
    """Retrieve the last 7 predictions for a specific child, sorted chronologically."""
    sql = """
        SELECT id, probability, risk_level, created_at
        FROM predictions
        WHERE child_id = %s AND user_id = %s
        ORDER BY created_at DESC
        LIMIT 7
    """
    connection = None
    cursor     = None
    try:
        connection = get_connection()
        cursor     = connection.cursor(dictionary=True)
        cursor.execute(sql, (child_id, user_id))
        records = cursor.fetchall()
        # Reverse to return in chronological order (oldest to newest)
        records.reverse()
        return records
    except Exception as e:
        print(f"[ERROR] get_child_weekly_trends failed: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

