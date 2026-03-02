from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PredictionRow:
    id: int
    created_at: str
    user_email: str
    file_name: str
    age: Optional[int]
    gender: Optional[str]
    name: Optional[str]
    risk_probability: float
    label: str
    confidence: float
    explanation: str


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                user_email TEXT NOT NULL,
                file_name TEXT NOT NULL,
                age INTEGER NULL,
                gender TEXT NULL,
                name TEXT NULL,
                risk_probability REAL NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                explanation TEXT NOT NULL
            )
            """
        )
        # Lightweight migration for older DBs created before the `name` column existed.
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        if "name" not in existing_cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN name TEXT NULL")
        conn.commit()


def _row_to_prediction(row: sqlite3.Row) -> PredictionRow:
    keys = set(row.keys())
    return PredictionRow(
        id=int(row["id"]),
        created_at=str(row["created_at"]),
        user_email=str(row["user_email"]),
        file_name=str(row["file_name"]),
        age=row["age"],
        gender=row["gender"],
        name=row["name"] if "name" in keys else None,
        risk_probability=float(row["risk_probability"]),
        label=str(row["label"]),
        confidence=float(row["confidence"]),
        explanation=str(row["explanation"]),
    )


def insert_prediction(
    *,
    db_path: str,
    user_email: str,
    file_name: str,
    age: Optional[int],
    gender: Optional[str],
    name: Optional[str],
    risk_probability: float,
    label: str,
    confidence: float,
    explanation: str,
) -> int:
    created_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        cur = conn.execute(
            """
            INSERT INTO predictions (
                created_at, user_email, file_name, age, gender, name,
                risk_probability, label, confidence, explanation
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                user_email,
                file_name,
                age,
                gender,
                name,
                float(risk_probability),
                label,
                float(confidence),
                explanation,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_prediction(
    db_path: str, prediction_id: int, user_email: Optional[str] = None
) -> Optional[PredictionRow]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if user_email:
            cur = conn.execute(
                "SELECT * FROM predictions WHERE id = ? AND user_email = ?",
                (int(prediction_id), str(user_email)),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM predictions WHERE id = ?",
                (int(prediction_id),),
            )
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_prediction(row)


def list_predictions(
    db_path: str, limit: Optional[int] = None, user_email: Optional[str] = None
) -> list[PredictionRow]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        params_list: list[object] = []
        sql = "SELECT * FROM predictions"
        if user_email:
            sql += " WHERE user_email = ?"
            params_list.append(str(user_email))
        sql += " ORDER BY id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params_list.append(int(limit))
        cur = conn.execute(sql, tuple(params_list))
        return [_row_to_prediction(r) for r in cur.fetchall()]
