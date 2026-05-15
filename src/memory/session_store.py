"""SQLite-backed store for persisting and retrieving AgentState sessions across runs."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.logging_config import get_logger
from src.schemas.models import AgentState

logger = get_logger(__name__)


class SessionStore:
    """Persist and retrieve AgentState objects in a local SQLite database."""

    db_path: str

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """)

    def save(self, state: AgentState) -> None:
        """Insert or update a session record for the given state."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (state.session_id, state.model_dump_json(), now, now),
            )
        logger.info("session_saved", session_id=state.session_id)

    def load(self, session_id: str) -> AgentState | None:
        """Return the AgentState for the given session ID, or None if not found."""
        with self._conn() as conn:
            row = conn.execute("SELECT state_json FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        logger.info("session_loaded", session_id=session_id)
        return AgentState.model_validate_json(row["state_json"])

    def delete(self, session_id: str) -> None:
        """Remove the session record with the given ID."""
        with self._conn() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        logger.info("session_deleted", session_id=session_id)

    def list_sessions(self) -> list[str]:
        """Return all session IDs ordered by most recently updated."""
        with self._conn() as conn:
            rows = conn.execute("SELECT session_id FROM sessions ORDER BY updated_at DESC").fetchall()
        return [str(row["session_id"]) for row in rows]
