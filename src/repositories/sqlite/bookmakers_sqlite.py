from __future__ import annotations

import sqlite3
from typing import Optional

from ..bookmakers import Bookmaker, BookmakersRepo


class BookmakersRepoSqlite(BookmakersRepo):
    """SQLite implementation of :class:`BookmakersRepo`."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bookmakers (
                bookmaker_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def get_by_id(self, bookmaker_id: int) -> Optional[Bookmaker]:
        cur = self._conn.execute(
            "SELECT bookmaker_id, name FROM bookmakers WHERE bookmaker_id = ?",
            (bookmaker_id,),
        )
        row = cur.fetchone()
        if row:
            return Bookmaker(*row)
        return None

    def list_all(self, *, limit: int = 100, offset: int = 0) -> list[Bookmaker]:
        cur = self._conn.execute(
            "SELECT bookmaker_id, name FROM bookmakers LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [Bookmaker(*row) for row in cur.fetchall()]

    def insert(self, bookmaker: Bookmaker) -> int:
        if bookmaker.id is None:
            cur = self._conn.execute("INSERT INTO bookmakers (name) VALUES (?)", (bookmaker.name,))
        else:
            cur = self._conn.execute(
                "INSERT INTO bookmakers (bookmaker_id, name) VALUES (?, ?)",
                (bookmaker.id, bookmaker.name),
            )
        self._conn.commit()
        rowid = cur.lastrowid
        if rowid is None:
            raise RuntimeError("SQLite insert failed: no lastrowid (table: bookmakers)")
        return int(rowid)

    def update(self, bookmaker: Bookmaker) -> None:
        self._conn.execute(
            "UPDATE bookmakers SET name = ? WHERE bookmaker_id = ?",
            (bookmaker.name, bookmaker.id),
        )
        self._conn.commit()

    def delete(self, bookmaker_id: int) -> None:
        self._conn.execute("DELETE FROM bookmakers WHERE bookmaker_id = ?", (bookmaker_id,))
        self._conn.commit()
