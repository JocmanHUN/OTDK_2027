from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional

from ..matches import Match, MatchesRepo


class MatchesRepoSqlite(MatchesRepo):
    """SQLite implementation of :class:`MatchesRepo`.

    Example:
        >>> import sqlite3
        >>> from datetime import datetime
        >>> conn = sqlite3.connect(":memory:")
        >>> repo = MatchesRepoSqlite(conn)
        >>> match_id = repo.insert(Match(None, 1, 2024, datetime.now(), "A", "B"))
        >>> repo.get_by_id(match_id)
        Match(id=1, league_id=1, season=2024, date=..., home_team='A', away_team='B', real_result=None)
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER PRIMARY KEY,
                league_id INTEGER NOT NULL,
                season INTEGER NOT NULL,
                date DATETIME NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                real_result TEXT CHECK(real_result IN ('1','X','2')),
                FOREIGN KEY (league_id) REFERENCES leagues(league_id)
            )
            """
        )
        self._conn.commit()

    def get_by_id(self, match_id: int) -> Optional[Match]:
        cur = self._conn.execute(
            """
            SELECT match_id, league_id, season, date, home_team, away_team, real_result
            FROM matches
            WHERE match_id = ?
            """,
            (match_id,),
        )
        row = cur.fetchone()
        if row:
            # row: (match_id, league_id, season, date, home_team, away_team, real_result)
            # Convert date string to datetime if needed
            dt = row[3]
            if isinstance(dt, str):
                try:
                    dt_parsed = datetime.fromisoformat(dt)
                except ValueError:
                    dt_parsed = datetime.fromtimestamp(0)
            else:
                dt_parsed = dt
            return Match(row[0], row[1], row[2], dt_parsed, row[4], row[5], row[6])
        return None

    def list_by_league(
        self, league_id: int, season: int, *, limit: int = 100, offset: int = 0
    ) -> list[Match]:
        cur = self._conn.execute(
            """
            SELECT match_id, league_id, season, date, home_team, away_team, real_result
            FROM matches
            WHERE league_id = ? AND season = ?
            LIMIT ? OFFSET ?
            """,
            (league_id, season, limit, offset),
        )
        rows = cur.fetchall()
        out: list[Match] = []
        for r in rows:
            dt = r[3]
            if isinstance(dt, str):
                try:
                    dt_parsed = datetime.fromisoformat(dt)
                except ValueError:
                    dt_parsed = datetime.fromtimestamp(0)
            else:
                dt_parsed = dt
            out.append(Match(r[0], r[1], r[2], dt_parsed, r[4], r[5], r[6]))
        return out

    def insert(self, match: Match) -> int:
        if match.id is None:
            cur = self._conn.execute(
                """
                INSERT INTO matches (league_id, season, date, home_team, away_team, real_result)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    match.league_id,
                    match.season,
                    (
                        match.date.isoformat()
                        if hasattr(match, "date")
                        else datetime.now().isoformat()
                    ),
                    match.home_team,
                    match.away_team,
                    match.real_result,
                ),
            )
        else:
            cur = self._conn.execute(
                """
                INSERT INTO matches (match_id, league_id, season, date, home_team, away_team, real_result)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match.id,
                    match.league_id,
                    match.season,
                    (
                        match.date.isoformat()
                        if hasattr(match, "date")
                        else datetime.now().isoformat()
                    ),
                    match.home_team,
                    match.away_team,
                    match.real_result,
                ),
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
            "UPDATE matches SET real_result = ? WHERE match_id = ?",
            (real_result, match_id),
        )
        self._conn.commit()

    def delete(self, match_id: int) -> None:
        self._conn.execute("DELETE FROM matches WHERE match_id = ?", (match_id,))
        self._conn.commit()
