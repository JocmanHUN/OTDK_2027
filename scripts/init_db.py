from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path


def ensure_schema(conn: sqlite3.Connection) -> None:
    # Programmatic schema init via repos so it always matches code
    from src.repositories.sqlite.bookmakers_sqlite import BookmakersRepoSqlite
    from src.repositories.sqlite.leagues_sqlite import LeaguesRepoSqlite
    from src.repositories.sqlite.matches_sqlite import MatchesRepoSqlite
    from src.repositories.sqlite.odds_sqlite import OddsRepoSqlite
    from src.repositories.sqlite.predictions_sqlite import PredictionsRepoSqlite

    LeaguesRepoSqlite(conn)
    BookmakersRepoSqlite(conn)
    MatchesRepoSqlite(conn)
    OddsRepoSqlite(conn)
    PredictionsRepoSqlite(conn)


def main() -> int:
    # Ensure project root (containing 'src') is importable
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    parser = argparse.ArgumentParser(description="Initialize SQLite database schema")
    parser.add_argument(
        "--db",
        default=os.path.join("data", "odtk.sqlite3"),
        help="Path to SQLite DB file (will be created if missing)",
    )
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
    finally:
        conn.close()

    print(f"Initialized schema at: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
