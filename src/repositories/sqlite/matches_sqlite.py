from __future__ import annotations

import sqlite3
from typing import Optional

from ..matches import Match, MatchesRepo


class MatchesRepoSqlite(MatchesRepo):
    """SQLite implementation of :class:`MatchesRepo`.

    Example:
        >>> import sqlite3
        >>> conn = sqlite3.connect(":memory:")
        >>> repo = MatchesRepoSqlite(conn)
        >>> match_id = repo.insert(Match(None, 1, 2024, "A", "B"))
        >>> repo.get_by_id(match_id)
        Match(id=1, league_id=1, season=2024, home_team='A', away_team='B', real_result=None)
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                league_id INTEGER NOT NULL,
                season INTEGER NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                real_result TEXT
            )
            """
        )
        self._conn.commit()

    def get_by_id(self, match_id: int) -> Optional[Match]:
        cur = self._conn.execute(
            "SELECT id, league_id, season, home_team, away_team, real_result FROM matches WHERE id = ?",
            (match_id,),
        )
        row = cur.fetchone()
        if row:
            return Match(*row)
        return None

    def list_by_league(
        self, league_id: int, season: int, *, limit: int = 100, offset: int = 0
    ) -> list[Match]:
        cur = self._conn.execute(
            """
            SELECT id, league_id, season, home_team, away_team, real_result
            FROM matches
            WHERE league_id = ? AND season = ?
            LIMIT ? OFFSET ?
            """,
            (league_id, season, limit, offset),
        )
        return [Match(*row) for row in cur.fetchall()]

    def insert(self, match: Match) -> int:
        cur = self._conn.execute(
            "INSERT INTO matches (league_id, season, home_team, away_team, real_result) VALUES (?, ?, ?, ?, ?)",
            (match.league_id, match.season, match.home_team, match.away_team, match.real_result),
        )
        self._conn.commit()
        rowid = cur.lastrowid
        if rowid is None:
            raise RuntimeError("SQLite insert failed: no lastrowid (table: matches)")
        return int(rowid)

    def update_result(self, match_id: int, real_result: str) -> None:
        if real_result not in {"1", "X", "2"}:
            raise ValueError("real_result must be one of '1', 'X', '2'")
        self._conn.execute(
            "UPDATE matches SET real_result = ? WHERE id = ?", (real_result, match_id)
        )
        self._conn.commit()

    def delete(self, match_id: int) -> None:
        self._conn.execute("DELETE FROM matches WHERE id = ?", (match_id,))
        self._conn.commit()
