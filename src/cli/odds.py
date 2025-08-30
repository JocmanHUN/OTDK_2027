from __future__ import annotations

import argparse
from decimal import Decimal
from typing import Sequence

from src.application.services.odds_service import OddsService
from src.domain.entities.odds import best_of
from src.domain.value_objects.enums import Outcome


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch 1X2 odds for a fixture")
    p.add_argument("--fixture", type=int, required=True, help="Fixture ID to fetch odds for")
    p.add_argument("--best", action="store_true", help="Show best odds across bookmakers")
    return p


def _fmt_dec(x: Decimal) -> str:
    return format(x, "f")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    svc = OddsService()
    odds = svc.get_fixture_odds(args.fixture)

    if not odds:
        print("No odds found.")
        return 0

    if args.best:
        bo = best_of(odds)
        print(
            "Best HOME:",
            f"bookmaker={bo[Outcome.HOME][0]} odd={_fmt_dec(bo[Outcome.HOME][1])}",
        )
        print(
            "Best DRAW:",
            f"bookmaker={bo[Outcome.DRAW][0]} odd={_fmt_dec(bo[Outcome.DRAW][1])}",
        )
        print(
            "Best AWAY:",
            f"bookmaker={bo[Outcome.AWAY][0]} odd={_fmt_dec(bo[Outcome.AWAY][1])}",
        )
        return 0

    # Default: list all odds
    for o in odds:
        print(
            f"fixture={o.fixture_id} bookmaker={o.bookmaker_id} "
            f"H={_fmt_dec(o.home)} D={_fmt_dec(o.draw)} A={_fmt_dec(o.away)}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
