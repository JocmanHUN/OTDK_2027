from __future__ import annotations

import sqlite3
from typing import Optional

from ..leagues import League, LeaguesRepo


class LeaguesRepoSqlite(LeaguesRepo):
    """SQLite implementation of :class:`LeaguesRepo`."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS leagues (
                league_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                country TEXT
            )
            """
        )
        self._conn.commit()

    def get_by_id(self, league_id: int) -> Optional[League]:
        cur = self._conn.execute(
            "SELECT league_id, name, country FROM leagues WHERE league_id = ?",
            (league_id,),
        )
        row = cur.fetchone()
        if row:
            return League(*row)
        return None

    def list_all(self, *, limit: int = 100, offset: int = 0) -> list[League]:
        cur = self._conn.execute(
            "SELECT league_id, name, country FROM leagues LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [League(*row) for row in cur.fetchall()]

    def insert(self, league: League) -> int:
        if league.id is None:
            cur = self._conn.execute(
                "INSERT INTO leagues (name, country) VALUES (?, ?)",
                (league.name, league.country),
            )
        else:
            cur = self._conn.execute(
                "INSERT INTO leagues (league_id, name, country) VALUES (?, ?, ?)",
                (league.id, league.name, league.country),
            )
        self._conn.commit()
        rowid = cur.lastrowid
        if rowid is None:
            raise RuntimeError("SQLite insert failed: no lastrowid (table: leagues)")
        return int(rowid)

    def update(self, league: League) -> None:
        self._conn.execute(
            "UPDATE leagues SET name = ?, country = ? WHERE league_id = ?",
            (league.name, league.country, league.id),
        )
        self._conn.commit()

    def delete(self, league_id: int) -> None:
        self._conn.execute(
            "DELETE FROM leagues WHERE league_id = ?",
            (league_id,),
        )
        self._conn.commit()
