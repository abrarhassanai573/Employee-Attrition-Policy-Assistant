import sqlite3
from datetime import datetime

DB_PATH = "history.db"


def init_db():
    """Creates the history table if it doesn't already exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            answer TEXT,
            provider TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_chat(question, answer, provider):
    """Saves a question/answer pair into SQLite."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO chat_history (timestamp, question, answer, provider) VALUES (?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question, answer, provider)
    )
    conn.commit()
    conn.close()


def fetch_history():
    """Returns all saved chat history, newest first."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, question, answer, provider FROM chat_history ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows
