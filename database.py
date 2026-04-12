import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path("./history.db")


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                filename      TEXT    NOT NULL,
                prediction    TEXT    NOT NULL,
                confidence    REAL    NOT NULL,
                gradcam_b64   TEXT,
                thumbnail_b64 TEXT
            )
        """)


def save_prediction(filename: str, prediction: str, confidence: float,
                    gradcam_b64: str, thumbnail_b64: str) -> int:
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO predictions
               (timestamp, filename, prediction, confidence, gradcam_b64, thumbnail_b64)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (datetime.utcnow().isoformat(timespec="seconds") + "Z",
             filename, prediction, round(confidence, 2), gradcam_b64, thumbnail_b64),
        )
        return cur.lastrowid


def get_history(limit: int = 50) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT id, timestamp, filename, prediction, confidence, thumbnail_b64 "
            "FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_prediction(pred_id: int) -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_prediction(pred_id: int) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    return cur.rowcount > 0


def clear_history():
    with _conn() as con:
        con.execute("DELETE FROM predictions")
