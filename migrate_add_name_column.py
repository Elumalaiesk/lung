import sqlite3
from pathlib import Path

db_path = Path(__file__).parent / "instance" / "predictions.sqlite3"

with sqlite3.connect(db_path) as conn:
    try:
        conn.execute("ALTER TABLE predictions ADD COLUMN name TEXT;")
        print("Column 'name' added successfully.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("Column 'name' already exists.")
        else:
            print("Error:", e)
