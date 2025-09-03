from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path


def ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def ensure_schema(conn: sqlite3.Connection) -> None:
    # Create tables if they don't exist yet
    from src.repositories.sqlite.leagues_sqlite import LeaguesRepoSqlite

    LeaguesRepoSqlite(conn)  # constructor runs CREATE TABLE IF NOT EXISTS


def seed_current_leagues(conn: sqlite3.Connection) -> tuple[int, int, int]:
    from src.application.services.leagues_service import LeaguesService
    from src.repositories.leagues import League
    from src.repositories.sqlite.leagues_sqlite import LeaguesRepoSqlite

    svc = LeaguesService()
    repo = LeaguesRepoSqlite(conn)

    items = svc.get_current_leagues()
    inserted = 0
    updated = 0
    skipped = 0

    for it in items:
        v = it.get("league_id")
        if isinstance(v, int):
            lid = v
        elif isinstance(v, str):
            try:
                lid = int(v)
            except ValueError:
                continue
        else:
            continue
        name = str(it.get("league_name"))
        country = it.get("country_name")
        existing = repo.get_by_id(lid)
        if existing is None:
            repo.insert(League(id=lid, name=name, country=country))
            inserted += 1
        else:
            if existing.name != name or existing.country != country:
                repo.update(League(id=lid, name=name, country=country))
                updated += 1
            else:
                skipped += 1

    return inserted, updated, skipped


def main(argv: list[str] | None = None) -> int:
    ensure_import_path()

    parser = argparse.ArgumentParser(description="Seed leagues table from API-FOOTBALL")
    parser.add_argument(
        "--db",
        default=os.path.join("data", "odtk.sqlite3"),
        help="Path to SQLite DB file",
    )
    args = parser.parse_args(argv)

    db_path = os.path.abspath(args.db)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
        ins, upd, skip = seed_current_leagues(conn)
    finally:
        conn.close()

    print(f"Leagues seeding done. inserted={ins} updated={upd} skipped={skip}")
    print(f"DB: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
