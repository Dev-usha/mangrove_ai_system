import bcrypt
import psycopg
import streamlit as st
from datetime import datetime
from psycopg.errors import UniqueViolation


def get_connection():
    return psycopg.connect(st.secrets["DATABASE_URL"])


def initialize_user_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id BIGSERIAL PRIMARY KEY,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP NOT NULL
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()


def hash_password(password):
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")


def verify_password(password, password_hash):
    return bcrypt.checkpw(
        password.encode("utf-8"),
        password_hash.encode("utf-8")
    )


def register_user(full_name, email, password):

    initialize_user_database()

    full_name = full_name.strip()
    email = email.strip().lower()

    if not full_name:
        return False, "Full name is required."

    if not email:
        return False, "Email is required."

    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    password_hash = hash_password(password)

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO users
            (full_name, email, password_hash, created_at)
            VALUES (%s, %s, %s, %s)
            """,
            (
                full_name,
                email,
                password_hash,
                datetime.now(),
            ),
        )

        conn.commit()

        return True, "Account created successfully."

    except UniqueViolation:

        conn.rollback()

        return False, "An account with this email already exists."

    finally:

        cursor.close()
        conn.close()


def login_user(email, password):

    initialize_user_database()

    email = email.strip().lower()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            full_name,
            email,
            password_hash,
            role
        FROM users
        WHERE email = %s
        """,
        (email,),
    )

    user = cursor.fetchone()

    cursor.close()
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
        "role": role,
    }

    return True, user_data, "Login successful."
