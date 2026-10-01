"""SQLite persistence for analyses and feedback (zero-config, single file)."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from backend.core import config

_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(config.DB_PATH, check_same_thread=False, isolation_level=None)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _lock, closing(_conn()) as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            mode TEXT NOT NULL,
            verdict TEXT NOT NULL,
            fake_probability REAL NOT NULL,
            confidence REAL NOT NULL,
            domain TEXT,
            snippet TEXT NOT NULL,
            latency_ms INTEGER NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT NOT NULL,
            rating TEXT NOT NULL,
            comment TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
        """)


def save_analysis(result: dict):
    with _lock, closing(_conn()) as c:
        c.execute(
            "INSERT OR REPLACE INTO analyses VALUES (?,?,?,?,?,?,?,?,?,?)",
            (result["id"], result["created_at"], result["mode"], result["verdict"],
             result["fake_probability"], result["confidence"], result["input"].get("domain"),
             result["input"]["text"][:160], result["latency_ms"], json.dumps(result)),
        )


def get_analysis(analysis_id: str) -> dict | None:
    with _lock, closing(_conn()) as c:
        row = c.execute("SELECT payload FROM analyses WHERE id=?", (analysis_id,)).fetchone()
    return json.loads(row["payload"]) if row else None


def history(limit: int = 20) -> list[dict]:
    with _lock, closing(_conn()) as c:
        rows = c.execute(
            "SELECT id, created_at, mode, verdict, fake_probability, confidence, domain, snippet "
            "FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def add_feedback(analysis_id: str, rating: str, comment: str | None):
    with _lock, closing(_conn()) as c:
        c.execute("INSERT INTO feedback (analysis_id, rating, comment, created_at) VALUES (?,?,?,?)",
                  (analysis_id, rating, comment, datetime.now(timezone.utc).isoformat()))


def usage() -> dict:
    with _lock, closing(_conn()) as c:
        total = c.execute("SELECT COUNT(*) n, AVG(latency_ms) l FROM analyses").fetchone()
        verdicts = c.execute("SELECT verdict, COUNT(*) n FROM analyses GROUP BY verdict").fetchall()
        fb = c.execute("SELECT rating, COUNT(*) n FROM feedback GROUP BY rating").fetchall()
        by_day = c.execute("SELECT substr(created_at,1,10) day, COUNT(*) n FROM analyses "
                           "GROUP BY day ORDER BY day DESC LIMIT 30").fetchall()
    fbm = {r["rating"]: r["n"] for r in fb}
    correct, incorrect = fbm.get("correct", 0), fbm.get("incorrect", 0)
    return {
        "analyses": int(total["n"] or 0),
        "verdicts": {r["verdict"]: r["n"] for r in verdicts},
        "avg_latency_ms": round(float(total["l"] or 0)),
        "feedback": {"total": correct + incorrect, "correct": correct, "incorrect": incorrect,
                     "agreement": (correct / (correct + incorrect)) if (correct + incorrect) else None},
        "by_day": [{"day": r["day"], "count": r["n"]} for r in reversed(by_day)],
    }
