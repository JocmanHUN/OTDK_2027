from __future__ import annotations

import sqlite3
from typing import Optional

from ..odds import Odds, OddsRepo


class OddsRepoSqlite(OddsRepo):
    """SQLite implementation of :class:`OddsRepo`.

    Example:
        >>> conn = sqlite3.connect(":memory:")
        >>> repo = OddsRepoSqlite(conn)
        >>> odds_id = repo.insert(Odds(None, 1, 1, 2.0, 3.0, 4.0))
        >>> repo.get_by_id(odds_id).home
        2.0
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS odds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER NOT NULL,
                bookmaker_id INTEGER NOT NULL,
                home REAL NOT NULL,
                draw REAL NOT NULL,
                away REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def get_by_id(self, odds_id: int) -> Optional[Odds]:
        cur = self._conn.execute(
            "SELECT id, match_id, bookmaker_id, home, draw, away FROM odds WHERE id = ?",
            (odds_id,),
        )
        row = cur.fetchone()
        if row:
            return Odds(*row)
        return None

    def list_by_match(self, match_id: int, *, limit: int = 100, offset: int = 0) -> list[Odds]:
        cur = self._conn.execute(
            """
            SELECT id, match_id, bookmaker_id, home, draw, away
            FROM odds
            WHERE match_id = ?
            LIMIT ? OFFSET ?
            """,
            (match_id, limit, offset),
        )
        return [Odds(*row) for row in cur.fetchall()]

    def insert(self, odds: Odds) -> int:
        cur = self._conn.execute(
            "INSERT INTO odds (match_id, bookmaker_id, home, draw, away) VALUES (?, ?, ?, ?, ?)",
            (odds.match_id, odds.bookmaker_id, odds.home, odds.draw, odds.away),
        )
        self._conn.commit()
        rowid = cur.lastrowid
        if rowid is None:
            raise RuntimeError("SQLite insert failed: no lastrowid (table: odds)")
        return int(rowid)

    def update(self, odds: Odds) -> None:
        self._conn.execute(
            "UPDATE odds SET match_id = ?, bookmaker_id = ?, home = ?, draw = ?, away = ? WHERE id = ?",
            (odds.match_id, odds.bookmaker_id, odds.home, odds.draw, odds.away, odds.id),
        )
        self._conn.commit()

    def delete(self, odds_id: int) -> None:
        self._conn.execute("DELETE FROM odds WHERE id = ?", (odds_id,))
        self._conn.commit()

    def best_odds(self, match_id: int) -> dict[str, tuple[int, float]]:
        """Return the best (bookmaker_id, odd) per outcome key '1', 'X', '2'."""
        cur = self._conn.execute(
            "SELECT bookmaker_id, home, draw, away FROM odds WHERE match_id = ?",
            (match_id,),
        )
        best: dict[str, tuple[int, float]] = {}
        for bookmaker_id, home, draw, away in cur.fetchall():
            if "1" not in best or home > best["1"][1]:
                best["1"] = (bookmaker_id, float(home))
            if "X" not in best or draw > best["X"][1]:
                best["X"] = (bookmaker_id, float(draw))
            if "2" not in best or away > best["2"][1]:
                best["2"] = (bookmaker_id, float(away))
        return best
