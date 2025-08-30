from __future__ import annotations

import argparse
from datetime import datetime
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from src.application.services.fixtures_service import FixturesService


def _format_rows(rows: Iterable[Mapping[str, object]], out_tz: str = "UTC") -> str:
    lines: list[str] = []
    for r in rows:
        dt = r.get("date_utc")
        # Convert UTC datetime to desired timezone for display, if possible
        if isinstance(dt, datetime):
            try:
                dt = dt.astimezone(ZoneInfo(out_tz))
            except Exception:
                # Fallback: leave as-is if timezone conversion fails
                pass
        home = r.get("home_name") or "?"
        away = r.get("away_name") or "?"
        status = r.get("status") or "?"
        league_id = r.get("league_id")
        season = r.get("season")
        fxid = r.get("fixture_id")
        lines.append(
            f"{dt} | {home} vs {away} [{status}] (league={league_id}, season={season}, id={fxid})"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="List daily fixtures (UTC output)")
    p.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
    p.add_argument("--league", type=int, help="Filter by league id")
    p.add_argument("--season", type=int, help="Filter by season year")
    p.add_argument(
        "--timezone",
        default="Europe/Budapest",
        help="Input timezone for the given date (default: Europe/Budapest)",
    )
    # No persistent debug/raw options — keep CLI minimal
    p.add_argument(
        "--out-timezone",
        default=None,
        help=(
            "Timezone for displaying kickoff times. By default uses the --timezone value "
            "(e.g., Europe/Budapest)."
        ),
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    svc = FixturesService()
    rows = svc.get_daily_fixtures(
        args.date,
        league_id=args.league,
        season=args.season,
        tz_name=args.timezone,
    )

    if not rows:
        print("No fixtures found.")
    else:
        display_tz = str(args.out_timezone) if args.out_timezone else str(args.timezone)
        print(_format_rows(rows, out_tz=display_tz))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
