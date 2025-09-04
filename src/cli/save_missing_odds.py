from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path
from typing import Sequence

from src.application.services.odds_service import OddsService
from src.repositories.bookmakers import Bookmaker
from src.repositories.odds import Odds as RepoOdds
from src.repositories.sqlite.bookmakers_sqlite import BookmakersRepoSqlite
from src.repositories.sqlite.odds_sqlite import OddsRepoSqlite

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "app.db"
MIGRATION_FILE = Path(__file__).resolve().parents[2] / "migrations" / "V1__base.sql"


def ensure_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(Path(db_path).parent, exist_ok=True)
    need_migrate = not os.path.exists(db_path)
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    if need_migrate:
        with open(MIGRATION_FILE, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
    return conn


def find_matches_missing_odds(conn: sqlite3.Connection) -> list[int]:
    cur = conn.execute(
        """
        SELECT DISTINCT p.match_id
        FROM predictions p
        LEFT JOIN odds o ON o.match_id = p.match_id
        WHERE o.match_id IS NULL
        ORDER BY p.match_id
        """
    )
    return [int(r[0]) for r in cur.fetchall()]


def persist_odds_for_match(conn: sqlite3.Connection, fixture_id: int, svc: OddsService) -> int:
    brepo = BookmakersRepoSqlite(conn)
    orepo = OddsRepoSqlite(conn)

    # Bookmaker names first (for nicer UI later)
    try:
        names = svc.get_fixture_bookmakers(fixture_id)
    except Exception:
        names = {}
    for bid, name in names.items():
        try:
            if brepo.get_by_id(int(bid)) is None:
                brepo.insert(Bookmaker(id=int(bid), name=str(name)))
        except Exception:
            pass

    # Odds upsert
    try:
        odds_list = svc.get_fixture_odds(fixture_id)
    except Exception:
        return 0

    written = 0
    for o in odds_list:
        try:
            orepo.insert(
                RepoOdds(
                    id=None,
                    match_id=int(getattr(o, "fixture_id", fixture_id)),
                    bookmaker_id=int(getattr(o, "bookmaker_id")),
                    home=float(getattr(o, "home")),
                    draw=float(getattr(o, "draw")),
                    away=float(getattr(o, "away")),
                )
            )
            written += 1
        except Exception:
            # Ignore individual bad rows; schema-level checks guard correctness
            continue
    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Save odds from API for all matches that have predictions but no odds in the DB."
        )
    )
    p.add_argument("--db", default=str(DEFAULT_DB), help="Path to SQLite DB (default: data/app.db)")
    p.add_argument("--limit", type=int, default=None, help="Optional limit on number of matches")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    conn = ensure_db(args.db)
    try:
        matches = find_matches_missing_odds(conn)
        if args.limit is not None:
            matches = matches[: int(args.limit)]

        if not matches:
            print("No matches missing odds found.")
            return 0

        svc = OddsService()
        total_rows = 0
        for mid in matches:
            rows = persist_odds_for_match(conn, int(mid), svc)
            total_rows += rows
            print(f"match_id={mid}: saved {rows} odds rows")

        print(f"Done. Matches: {len(matches)}, odds rows written: {total_rows}")
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
