from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Sequence

from src.application.services.history_service import HistoryService
from src.application.services.prediction_pipeline import ContextBuilder, PredictionAggregatorImpl
from src.domain.entities.match import Match
from src.domain.value_objects.enums import MatchStatus
from src.domain.value_objects.ids import FixtureId, LeagueId
from src.models.poisson import PoissonModel


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Poisson prediction for a fixture ID")
    p.add_argument("--fixture", type=int, required=True, help="Fixture ID")
    p.add_argument("--league", type=int, required=True, help="League ID")
    p.add_argument("--season", type=int, required=True, help="Season year (e.g., 2024)")
    p.add_argument("--home", type=int, required=True, help="Home team ID")
    p.add_argument("--away", type=int, required=True, help="Away team ID")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Build modeling context from API-driven team stats
    history = HistoryService()
    ctx_builder = ContextBuilder(history=history)
    ctx = ctx_builder.build_from_meta(
        fixture_id=args.fixture,
        league_id=args.league,
        season=args.season,
        home_team_id=args.home,
        away_team_id=args.away,
    )

    # Minimal Match stub (names not required by model)
    match = Match(
        fixture_id=FixtureId(int(args.fixture)),
        league_id=LeagueId(int(args.league)),
        season=int(args.season),
        kickoff_utc=datetime.now(timezone.utc),
        home_name=str(args.home),
        away_name=str(args.away),
        status=MatchStatus.SCHEDULED,
    )

    model = PoissonModel()
    agg = PredictionAggregatorImpl()
    preds = agg.run_all([model], match, ctx)
    pred = preds[0]

    if pred.probs is None:
        print(f"SKIPPED: {pred.skip_reason}")
        return 0

    print(
        f"Poisson probabilities for fixture={args.fixture}:\n"
        f"  home: {pred.probs.home:.4f}\n"
        f"  draw: {pred.probs.draw:.4f}\n"
        f"  away: {pred.probs.away:.4f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
