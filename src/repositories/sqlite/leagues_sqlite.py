from __future__ import annotations

import sqlite3
from typing import Optional

from ..leagues import League, LeaguesRepo


class LeaguesRepoSqlite(LeaguesRepo):
    """SQLite implementation of :class:`LeaguesRepo`."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS leagues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def get_by_id(self, league_id: int) -> Optional[League]:
        cur = self._conn.execute(
            "SELECT id, name FROM leagues WHERE id = ?", (league_id,)
        )
        row = cur.fetchone()
        if row:
            return League(*row)
        return None

    def list_all(self, *, limit: int = 100, offset: int = 0) -> list[League]:
        cur = self._conn.execute(
            "SELECT id, name FROM leagues LIMIT ? OFFSET ?", (limit, offset)
        )
        return [League(*row) for row in cur.fetchall()]

    def insert(self, league: League) -> int:
        cur = self._conn.execute(
            "INSERT INTO leagues (name) VALUES (?)", (league.name,)
        )
        self._conn.commit()
        return cur.lastrowid

    def update(self, league: League) -> None:
        self._conn.execute(
            "UPDATE leagues SET name = ? WHERE id = ?",
            (league.name, league.id),
        )
        self._conn.commit()

    def delete(self, league_id: int) -> None:
        self._conn.execute("DELETE FROM leagues WHERE id = ?", (league_id,))
        self._conn.commit()
