"""
database/db.py
--------------
MySQL database connection module for the
ASD Meltdown Probability Predictor project.

Handles:
  - Loading environment variables from .env
  - Creating MySQL connections
  - Providing a reusable connection context manager
  - Testing the database connection
  - Graceful error handling for Render/cloud deployment

Usage:
  from database.db import get_connection, test_connection
"""

import os
import mysql.connector
from mysql.connector import Error, InterfaceError, DatabaseError
from dotenv import load_dotenv

# ── Load environment variables from .env file ─────────────────────────────────
# Works locally (reads .env file) and on Render (reads injected env vars).
# override=False ensures Render's real env vars take priority over any .env file.
load_dotenv(override=False)


# ── Database Configuration ────────────────────────────────────────────────────
# All sensitive credentials are read from environment variables.
# Never hardcode these values directly in the source code.

DB_CONFIG = {
    "host":             os.getenv("DB_HOST",     "localhost"),
    "user":             os.getenv("DB_USER",     "root"),
    "password":         os.getenv("DB_PASSWORD", ""),
    "database":         os.getenv("DB_NAME",     "asd_predictor"),
    "port":             int(os.getenv("DB_PORT", 3306)),

    # ── Connection stability settings ─────────────────────────────────────
    "connection_timeout":    10,      # seconds to wait for initial connection
    "autocommit":            False,   # explicit commits required (safer)
    "charset":               "utf8mb4",
    "collation":             "utf8mb4_unicode_ci",
    "use_pure":              True,    # pure Python connector (Render compatible)
    "raise_on_warnings":     False,
}


# ─────────────────────────────────────────────────────────────────────────────
# Core Connection Function
# ─────────────────────────────────────────────────────────────────────────────

def get_connection() -> mysql.connector.MySQLConnection:
    """
    Create and return a new MySQL database connection.

    Reads all credentials from environment variables via DB_CONFIG.
    Suitable for both local development and Render cloud deployment.

    Returns
    -------
    mysql.connector.MySQLConnection
        An active, open MySQL connection object.

    Raises
    ------
    ConnectionError
        If the connection cannot be established (wraps mysql.connector.Error).

    Example
    -------
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users")
        conn.close()
    """
    try:
        connection = mysql.connector.connect(**DB_CONFIG)

        if connection.is_connected():
            return connection

    except Error as e:
        # Provide a clear, actionable error message
        raise ConnectionError(
            f"[DB ERROR] Could not connect to MySQL database.\n"
            f"  Host     : {DB_CONFIG['host']}\n"
            f"  Database : {DB_CONFIG['database']}\n"
            f"  Reason   : {e}\n\n"
            f"  Check your .env file or Render environment variables."
        ) from e

# ─────────────────────────────────────────────────────────────────────────────
# Connection Health Check
# ─────────────────────────────────────────────────────────────────────────────

def test_connection() -> dict:
    """
    Test the database connection and return a status report.

    Returns a dict with 'success', 'message', and optionally
    'server_version' so it can be displayed in the Streamlit UI
    or checked in a startup health check.

    Returns
    -------
    dict
        {
          "success":        bool,
          "message":        str,
          "server_version": str | None,
          "database":       str | None,
          "host":           str,
        }

    Example
    -------
        result = test_connection()
        if result["success"]:
            print("Connected:", result["server_version"])
        else:
            print("Failed:", result["message"])
    """
    try:
        connection = get_connection()

        # Fetch server metadata to confirm the connection is live
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


# ─────────────────────────────────────────────────────────────────────────────
# Schema Initialiser — Creates Tables If They Don't Exist
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> bool:
    """
    Run the SQL schema to create all tables if they don't already exist.

    Reads schema from database/schema.sql and executes it.
    Safe to call on every startup — uses CREATE TABLE IF NOT EXISTS.

    Returns
    -------
    bool
        True if schema was applied successfully, False otherwise.
    """
    # Path to schema.sql relative to this file
    schema_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "schema.sql"
    )

    if not os.path.exists(schema_path):
        print(f"[WARN] schema.sql not found at: {schema_path}")
        return False

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            raw_sql = f.read()

        # Split on semicolons to execute each statement individually
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
        return True

    except Exception as e:
        print(f"[ERROR] Failed to initialise schema: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point — Run Connection Test Directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 50)
    print("  ASD Predictor — Database Connection Test")
    print("═" * 50)

    result = test_connection()

    if result["success"]:
        print(f"\n  ✅ Status  : {result['message']}")
        print(f"  🖥  Host    : {result['host']}")
        print(f"  🗄  Database: {result['database']}")
        print(f"  ⚙  MySQL   : {result['server_version']}")
    else:
        print(f"\n  ❌ Status  : Connection FAILED")
        print(f"  Reason    : {result['message']}")

    print("\n" + "═" * 50)