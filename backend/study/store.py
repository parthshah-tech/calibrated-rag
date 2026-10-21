"""Append-only SQLite log for the user study: who was assigned which condition, and every event.
There are deliberately no update or delete methods."""

from __future__ import annotations

import json
import random
import sqlite3
import threading
import time
from pathlib import Path

CONDITIONS = ("baseline", "badge", "badge_reformulation")

_COLS = "code, condition, config_hash, started_at"
_KEYS = ("code", "condition", "config_hash", "started_at")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS participants(
    code TEXT PRIMARY KEY, condition TEXT NOT NULL, config_hash TEXT NOT NULL,
    started_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS events(
    id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT NOT NULL, question_id TEXT,
    type TEXT NOT NULL, ts REAL NOT NULL, payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_code ON events(code, id);
"""


class StudyStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript(_SCHEMA)

    def assign(self, code: str, conditions=CONDITIONS, config_hash: str = "", now=None):
        """Balanced random assignment: pick among the least-filled conditions, with a choice
        that is deterministic given the code. Returns (condition, created); a code that was
        already assigned keeps its condition, so refreshing the page cannot change it."""
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT condition FROM participants WHERE code=?", (code,)
            ).fetchone()
            if row:
                return row[0], False
            counts = {
                c: self._db.execute(
                    "SELECT COUNT(*) FROM participants WHERE condition=?", (c,)
                ).fetchone()[0]
                for c in conditions
            }
            low = min(counts.values())
            pool = sorted(c for c, n in counts.items() if n == low)
            condition = random.Random(code).choice(pool)
            self._db.execute(
                f"INSERT INTO participants({_COLS}) VALUES(?,?,?,?)",
                (code, condition, config_hash, now if now is not None else time.time()),
            )
            return condition, True

    def participant(self, code: str) -> dict | None:
        with self._lock:
            r = self._db.execute(
                f"SELECT {_COLS} FROM participants WHERE code=?", (code,)
            ).fetchone()
        return dict(zip(_KEYS, r, strict=True)) if r else None

    def participants(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_COLS} FROM participants ORDER BY started_at, code"
            ).fetchall()
        return [dict(zip(_KEYS, r, strict=True)) for r in rows]

    def log(self, code: str, question_id: str | None, type_: str, payload: dict, ts=None) -> int:
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO events(code, question_id, type, ts, payload) VALUES(?,?,?,?,?)",
                (
                    code,
                    question_id,
                    type_,
                    ts if ts is not None else time.time(),
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            return cur.lastrowid

    def serialize(self) -> bytes:
        """A consistent copy of the whole database, safe to take while the study is running."""
        with self._lock:
            return self._db.serialize()

    def events(self, code: str | None = None, question_id: str | None = None) -> list[dict]:
        sql, args = "SELECT id, code, question_id, type, ts, payload FROM events", []
        clauses = []
        if code is not None:
            clauses.append("code=?")
            args.append(code)  # noqa: E702
        if question_id is not None:
            clauses.append("question_id=?")
            args.append(question_id)  # noqa: E702
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY id", args).fetchall()
        return [
            {
                "id": r[0],
                "code": r[1],
                "question_id": r[2],
                "type": r[3],
                "ts": r[4],
                "payload": json.loads(r[5]),
            }
            for r in rows
        ]
