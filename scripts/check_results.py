import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.application.services.history_service import HistoryService  # noqa: E402
from src.infrastructure.api_football_client import APIError  # noqa: E402

DB_PATH = ROOT / "data" / "app.db"


@dataclass
class MatchRow:
    match_id: int
    home_team: str
    away_team: str
    stored_result: str
    date_utc: str


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    return conn


def _load_finished_matches(conn: sqlite3.Connection) -> List[MatchRow]:
    rows = conn.execute(
        """
        SELECT match_id, home_team, away_team, real_result, date
        FROM matches
        WHERE real_result IS NOT NULL
        ORDER BY date DESC
        """
    ).fetchall()
    result: List[MatchRow] = []
    for r in rows:
        try:
            result.append(
                MatchRow(
                    match_id=int(r["match_id"]),
                    home_team=str(r["home_team"]),
                    away_team=str(r["away_team"]),
                    stored_result=str(r["real_result"]),
                    date_utc=str(r["date"]),
                )
            )
        except Exception:
            continue
    return result


def check_results(sleep_sec: float = 0.4, limit: Optional[int] = None) -> None:
    if not DB_PATH.exists():
        print(f"DB not found at {DB_PATH}")
        return

    conn = _connect()
    matches = _load_finished_matches(conn)
    if limit is not None:
        matches = matches[: int(limit)]

    svc = HistoryService()
    mismatches: List[MatchRow] = []
    unknown: List[MatchRow] = []

    for idx, m in enumerate(matches, 1):
        try:
            api_res = svc.get_fixture_result_label(m.match_id)
        except APIError as exc:
            print(f"[STOP] API limit/error at match {m.match_id}: {exc}")
            break
        except Exception as exc:
            print(f"[WARN] match {m.match_id} ({m.home_team}-{m.away_team}): {exc}")
            api_res = None

        if api_res is None:
            unknown.append(m)
        elif api_res != m.stored_result:
            mismatches.append(m)

        if sleep_sec > 0:
            time.sleep(sleep_sec)

        if idx % 50 == 0:
            print(f"Checked {idx} matches... mismatches={len(mismatches)}, unknown={len(unknown)}")

    print("=== Check complete ===")
    print(f"Total checked: {len(matches)}")
    print(f"Mismatches: {len(mismatches)}")
    print(f"Unknown (no result from API): {len(unknown)}")

    if mismatches:
        print("\n-- Mismatching results --")
        for m in mismatches:
            print(
                f"{m.match_id} | {m.date_utc} | {m.home_team} - {m.away_team} | "
                f"db={m.stored_result} vs api=DIFF"
            )
    if unknown:
        print("\n-- No result returned by API --")
        for m in unknown:
            print(f"{m.match_id} | {m.date_utc} | {m.home_team} - {m.away_team}")


if __name__ == "__main__":
    check_results()
