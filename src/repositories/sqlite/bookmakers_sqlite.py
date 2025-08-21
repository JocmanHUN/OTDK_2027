from __future__ import annotations

import sqlite3
from typing import Optional

from ..bookmakers import Bookmaker, BookmakersRepo


class BookmakersRepoSqlite(BookmakersRepo):
    """SQLite implementation of :class:`BookmakersRepo`."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bookmakers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def get_by_id(self, bookmaker_id: int) -> Optional[Bookmaker]:
        cur = self._conn.execute("SELECT id, name FROM bookmakers WHERE id = ?", (bookmaker_id,))
        row = cur.fetchone()
        if row:
            return Bookmaker(*row)
        return None

    def list_all(self, *, limit: int = 100, offset: int = 0) -> list[Bookmaker]:
        cur = self._conn.execute(
            "SELECT id, name FROM bookmakers LIMIT ? OFFSET ?", (limit, offset)
        )
        return [Bookmaker(*row) for row in cur.fetchall()]

    def insert(self, bookmaker: Bookmaker) -> int:
        cur = self._conn.execute("INSERT INTO bookmakers (name) VALUES (?)", (bookmaker.name,))
        self._conn.commit()
        return cur.lastrowid

    def update(self, bookmaker: Bookmaker) -> None:
        self._conn.execute(
            "UPDATE bookmakers SET name = ? WHERE id = ?",
            (bookmaker.name, bookmaker.id),
        )
        self._conn.commit()

    def delete(self, bookmaker_id: int) -> None:
        self._conn.execute("DELETE FROM bookmakers WHERE id = ?", (bookmaker_id,))
        self._conn.commit()
