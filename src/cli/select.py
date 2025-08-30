from __future__ import annotations

import argparse
from typing import Iterable, Sequence

from src.application.services.selection_pipeline import (
    FixtureShort,
    LeagueSeason,
    SelectionPipeline,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="League & match selection pipeline")
    p.add_argument("--date", required=True, help="Day to list (YYYY-MM-DD)")
    p.add_argument(
        "--timezone",
        default="Europe/Budapest",
        help="Timezone for daily fixtures (default: Europe/Budapest)",
    )
    p.add_argument("--leagues-only", action="store_true", help="Print supported leagues and exit")
    p.add_argument(
        "--with-odds",
        action="store_true",
        help="Filter fixtures to those that have 1X2 odds",
    )
    p.add_argument(
        "--not-finished",
        action="store_true",
        help="Only keep fixtures that are not finished (status != FT)",
    )
    p.add_argument(
        "--print-fixtures",
        action="store_true",
        help="Print fixture lines instead of only the summary",
    )
    return p


def _fmt_leagues(rows: Iterable[LeagueSeason]) -> str:
    return "\n".join(f"league={r['league_id']} season={r['season_year']}" for r in rows)


def _fmt_fixtures(rows: Iterable[FixtureShort]) -> str:
    out: list[str] = []
    for r in rows:
        out.append(
            f"fixture={r['fixture_id']} league={r['league_id']} season={r['season_year']} "
            f"home={r.get('home_id')} away={r.get('away_id')}"
        )
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    pipe = SelectionPipeline.default()

    leagues = pipe.list_supported_leagues()
    if args.leagues_only:
        print(_fmt_leagues(leagues))
        return 0

    fixtures = pipe.list_daily_matches(args.date, leagues, tz_name=str(args.timezone))
    # Optional: filter by status (not finished)
    if args.not_finished:
        fixtures = [f for f in fixtures if f.get("status") != "FT"]
    filtered = pipe.filter_matches_with_1x2_odds(fixtures) if args.with_odds else fixtures

    summary = f"leagues={len(leagues)} fixtures={len(fixtures)}"
    if args.with_odds:
        summary += f" fixtures_with_odds={len(filtered)}"
    print(summary)
    if args.print_fixtures:
        print(_fmt_fixtures(filtered))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
