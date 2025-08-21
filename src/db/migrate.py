"""Simple SQLite migration runner."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent.parent
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
DB_PATH = ROOT / "otdk.db"


def applied_versions(cursor: sqlite3.Cursor) -> set[str]:
    cursor.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY)")
    rows = cursor.execute("SELECT version FROM schema_migrations").fetchall()
    return {row[0] for row in rows}


def available_migrations() -> Iterable[tuple[str, Path]]:
    pattern = re.compile(r"V(\d+)__.+\.sql$")
    for path in sorted(MIGRATIONS_DIR.glob("V*__*.sql")):
        match = pattern.match(path.name)
        if match:
            yield match.group(1), path


def run_migrations() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        done = applied_versions(cursor)
        for version, path in available_migrations():
            if version in done:
                continue
            sql = path.read_text()
            cursor.executescript(sql)
            cursor.execute("INSERT INTO schema_migrations (version) VALUES (?)", (version,))
            conn.commit()


if __name__ == "__main__":
    run_migrations()
