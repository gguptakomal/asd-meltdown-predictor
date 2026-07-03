"""
project/database.py
-------------------
MySQL database connection module for the Flask application.
"""

import os
import mysql.connector
from mysql.connector import Error, InterfaceError
from dotenv import load_dotenv

# Resolve project root path and load env variables
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"), override=False)

DB_CONFIG = {
    "host":             os.getenv("DB_HOST",     "localhost"),
    "user":             os.getenv("DB_USER",     "root"),
    "password":         os.getenv("DB_PASSWORD", ""),
    "database":         os.getenv("DB_NAME",     "asd_predictor"),
    "port":             int(os.getenv("DB_PORT", 3306)),
    
    # Stability settings
    "connection_timeout":    10,
    "autocommit":            False,
    "charset":               "utf8mb4",
    "collation":             "utf8mb4_unicode_ci",
    "use_pure":              True,
    "raise_on_warnings":     False,
}

def get_connection() -> mysql.connector.MySQLConnection:
    """Create and return a new MySQL database connection."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        raise ConnectionError(
            f"[DB ERROR] Could not connect to MySQL database.\n"
            f"  Host     : {DB_CONFIG['host']}\n"
            f"  Database : {DB_CONFIG['database']}\n"
            f"  Reason   : {e}\n"
        ) from e

def test_connection() -> dict:
    """Test the database connection and return a status report."""
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT VERSION(), DATABASE()")
        version, db_name = cursor.fetchone()
        cursor.close()
        connection.close()

        return {
            "success":        True,
            "message":        "Database connection successful.",
            "server_version": version,
            "database":       db_name,
            "host":           DB_CONFIG["host"],
        }
    except (ConnectionError, InterfaceError, Error) as e:
        return {
            "success":        False,
            "message":        str(e),
            "server_version": None,
            "database":       None,
            "host":           DB_CONFIG["host"],
        }

def init_db() -> bool:
    """Initialise database tables using database/schema.sql."""
    schema_path = os.path.join(ROOT, "database", "schema.sql")

    if not os.path.exists(schema_path):
        print(f"[WARN] schema.sql not found at: {schema_path}")
        return False

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            raw_sql = f.read()

        statements = [
            stmt.strip()
            for stmt in raw_sql.split(";")
            if stmt.strip()
        ]

        connection = get_connection()
        cursor = connection.cursor()

        for statement in statements:  
            cursor.execute(statement)

        connection.commit()
        cursor.close()
        connection.close()

        print("[INFO] Database schema initialised successfully.")
        run_migrations()
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialise schema: {e}")
        return False

def run_migrations():
    """Add columns dynamically if they are missing in the existing database."""
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()
        
        # Check child_profiles table for 'address' column
        try:
            cursor.execute("ALTER TABLE child_profiles ADD COLUMN address VARCHAR(255) NULL")
            connection.commit()
            print("[INFO] Migrated child_profiles: added 'address' column.")
        except Error as e:
            # Error 1060 means column already exists
            if e.errno == 1060:
                pass
            else:
                print(f"[WARN] Migration failed for child_profiles address: {e}")
                
        # Check predictions table for 'notes' column
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN notes TEXT NULL")
            connection.commit()
            print("[INFO] Migrated predictions: added 'notes' column.")
        except Error as e:
            # Error 1060 means column already exists
            if e.errno == 1060:
                pass
            else:
                print(f"[WARN] Migration failed for predictions notes: {e}")
                
    except Exception as e:
        print(f"[ERROR] Run migrations failed: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

