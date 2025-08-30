from __future__ import annotations

import argparse
from typing import Iterable, List, Mapping, Sequence

from src.application.services.leagues_service import LeaguesService


def _format_rows(rows: Iterable[Mapping[str, object]]) -> str:
    out_lines: List[str] = []
    for r in rows:
        year = r.get("season_year")
        name = r.get("league_name")
        country = r.get("country_name") or "-"
        lid = r.get("league_id")
        out_lines.append(f"{year}: {name} ({country}) [id={lid}]")
    return "\n".join(out_lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="List API-FOOTBALL leagues with odds+stats coverage")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--current", action="store_true", help="List current-season leagues")
    g.add_argument(
        "--season", type=int, metavar="YEAR", help="List leagues for a specific season year"
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    svc = LeaguesService()

    if args.current:
        rows = svc.get_current_leagues()
    else:
        rows = svc.get_leagues_for_season(int(args.season))

    if not rows:
        print("No leagues with odds+stats coverage found.")
    else:
        print(_format_rows(rows))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
