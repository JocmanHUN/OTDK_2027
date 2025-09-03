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
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS odds (
                odds_id INTEGER PRIMARY KEY,
                match_id INTEGER NOT NULL,
                bookmaker_id INTEGER NOT NULL,
                odds_home REAL NOT NULL,
                odds_draw REAL NOT NULL,
                odds_away REAL NOT NULL,
                FOREIGN KEY (match_id) REFERENCES matches(match_id),
                FOREIGN KEY (bookmaker_id) REFERENCES bookmakers(bookmaker_id)
            )
            """
        )
        # Unique pair to avoid duplicate odds rows per bookmaker per match
        try:
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_odds_match_bookmaker ON odds(match_id, bookmaker_id)"
            )
        except sqlite3.IntegrityError:
            # Deduplicate by keeping the latest odds per (match_id, bookmaker_id)
            self._conn.execute(
                """
                DELETE FROM odds
                WHERE odds_id NOT IN (
                    SELECT MAX(odds_id) FROM odds GROUP BY match_id, bookmaker_id
                )
                """
            )
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_odds_match_bookmaker ON odds(match_id, bookmaker_id)"
            )
        self._conn.commit()

    def get_by_id(self, odds_id: int) -> Optional[Odds]:
        cur = self._conn.execute(
            "SELECT odds_id, match_id, bookmaker_id, odds_home, odds_draw, odds_away FROM odds WHERE odds_id = ?",
            (odds_id,),
        )
        row = cur.fetchone()
        if row:
            # Map DB column names to dataclass fields
            return Odds(row[0], row[1], row[2], row[3], row[4], row[5])
        return None

    def list_by_match(self, match_id: int, *, limit: int = 100, offset: int = 0) -> list[Odds]:
        cur = self._conn.execute(
            """
            SELECT odds_id, match_id, bookmaker_id, odds_home, odds_draw, odds_away
            FROM odds
            WHERE match_id = ?
            LIMIT ? OFFSET ?
            """,
            (match_id, limit, offset),
        )
        return [Odds(r[0], r[1], r[2], r[3], r[4], r[5]) for r in cur.fetchall()]

    def insert(self, odds: Odds) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO odds (match_id, bookmaker_id, odds_home, odds_draw, odds_away)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(match_id, bookmaker_id) DO UPDATE SET
                odds_home=excluded.odds_home,
                odds_draw=excluded.odds_draw,
                odds_away=excluded.odds_away
            """,
            (odds.match_id, odds.bookmaker_id, odds.home, odds.draw, odds.away),
        )
        self._conn.commit()
        rowid = cur.lastrowid
        if rowid is None:
            raise RuntimeError("SQLite insert failed: no lastrowid (table: odds)")
        return int(rowid)

    def update(self, odds: Odds) -> None:
        self._conn.execute(
            "UPDATE odds SET match_id = ?, bookmaker_id = ?, odds_home = ?, odds_draw = ?, odds_away = ? WHERE odds_id = ?",
            (odds.match_id, odds.bookmaker_id, odds.home, odds.draw, odds.away, odds.id),
        )
        self._conn.commit()

    def delete(self, odds_id: int) -> None:
        self._conn.execute("DELETE FROM odds WHERE odds_id = ?", (odds_id,))
        self._conn.commit()

    def best_odds(self, match_id: int) -> dict[str, tuple[int, float]]:
        """Return the best (bookmaker_id, odd) per outcome key '1', 'X', '2'."""
        cur = self._conn.execute(
            "SELECT bookmaker_id, odds_home, odds_draw, odds_away FROM odds WHERE match_id = ?",
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
