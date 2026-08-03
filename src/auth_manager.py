import os
import sqlite3
import bcrypt
from datetime import datetime


DB_DIR = "database"
DB_PATH = os.path.join(DB_DIR, "mangroveai_users.db")


def get_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def initialize_user_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


def hash_password(password):
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(password, password_hash):
    password_bytes = password.encode("utf-8")
    hash_bytes = password_hash.encode("utf-8")
    return bcrypt.checkpw(password_bytes, hash_bytes)


def register_user(full_name, email, password):
    initialize_user_database()

    full_name = full_name.strip()
    email = email.strip().lower()

    if not full_name:
        return False, "Full name is required."

    if not email:
        return False, "Email is required."

    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    password_hash = hash_password(password)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO users (full_name, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (full_name, email, password_hash, created_at)
        )

        conn.commit()
        conn.close()

        return True, "Account created successfully. You can now log in."

    except sqlite3.IntegrityError:
        conn.close()
        return False, "An account with this email already exists."


def login_user(email, password):
    initialize_user_database()

    email = email.strip().lower()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, full_name, email, password_hash, role
        FROM users
        WHERE email = ?
        """,
        (email,)
    )

    user = cursor.fetchone()
    conn.close()

    if user is None:
        return False, None, "No account found with this email."

    user_id, full_name, email, password_hash, role = user

    if not verify_password(password, password_hash):
        return False, None, "Incorrect password."

    user_data = {
        "user_id": user_id,
        "full_name": full_name,
        "email": email,
        "role": role
    }

    return True, user_data, "Login successful."