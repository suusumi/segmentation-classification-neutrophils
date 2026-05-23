"""SQLite-backed analysis metadata repository."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.services.errors import AnalysisNotFoundError


class SQLiteAnalysisRepository:
    """Persist analysis records in a local SQLite database."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    input_filename TEXT NOT NULL,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create(self, analysis_id: str, input_filename: str, status: str = "created") -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO analyses (
                    id,
                    status,
                    input_filename,
                    result_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (analysis_id, status, input_filename, None, now, now),
            )

    def update_result(self, analysis_id: str, status: str, result: dict[str, Any]) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE analyses
                SET status = ?, result_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, json.dumps(result, ensure_ascii=False), now, analysis_id),
            )
            if cursor.rowcount == 0:
                raise AnalysisNotFoundError(f"Analysis not found: {analysis_id}")

    def update_status(self, analysis_id: str, status: str) -> None:
        """Update analysis status without changing its result payload."""

        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE analyses
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, now, analysis_id),
            )
            if cursor.rowcount == 0:
                raise AnalysisNotFoundError(f"Analysis not found: {analysis_id}")

    def get(self, analysis_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM analyses WHERE id = ?",
                (analysis_id,),
            ).fetchone()
        if row is None:
            raise AnalysisNotFoundError(f"Analysis not found: {analysis_id}")

        payload = dict(row)
        result_json = payload.get("result_json")
        payload["result"] = json.loads(result_json) if result_json else None
        payload.pop("result_json", None)
        return payload
