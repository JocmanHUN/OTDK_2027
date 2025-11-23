from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence, cast

from src.application.services.history_service import HistoryService
from src.application.services.prediction_pipeline import ContextBuilder, PredictionAggregatorImpl
from src.application.services.selection_pipeline import SelectionPipeline
from src.domain.entities.match import Match
from src.domain.value_objects.enums import MatchStatus
from src.domain.value_objects.ids import FixtureId, LeagueId
from src.models import default_models


def _serialize_prediction(pred: Any) -> Mapping[str, Any]:
    base = {
        "fixture_id": int(pred.fixture_id),
        "model": str(pred.model.value if hasattr(pred.model, "value") else pred.model),
        "version": str(pred.version),
        "status": str(pred.status.value if hasattr(pred.status, "value") else pred.status),
        "computed_at_utc": pred.computed_at_utc.isoformat(),
    }
    if pred.probs is not None:
        base["probs"] = {"home": pred.probs.home, "draw": pred.probs.draw, "away": pred.probs.away}
    if pred.skip_reason:
        base["skip_reason"] = pred.skip_reason
    return base


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Quickly probe a few fixtures and print model outputs (no DB writes)."
    )
    p.add_argument(
        "--date", default=datetime.now(timezone.utc).date().isoformat(), help="YYYY-MM-DD"
    )
    p.add_argument(
        "--league-id",
        type=int,
        help="Optional league id to limit fixture fetch (reduces API calls).",
    )
    p.add_argument(
        "--limit", type=int, default=10, help="How many fixtures to probe (default: 10)."
    )
    p.add_argument("--with-odds", action="store_true", help="Only include fixtures with 1X2 odds.")
    p.add_argument("--json", action="store_true", help="Output JSON array instead of text.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    selector = SelectionPipeline.default()
    leagues = selector.list_supported_leagues()
    if args.league_id is not None:
        leagues = [ls for ls in leagues if int(ls["league_id"]) == int(args.league_id)]
    fixtures = selector.list_daily_matches(args.date, leagues)
    if args.with_odds:
        fixtures = selector.filter_matches_with_1x2_odds(fixtures)

    if args.limit and args.limit > 0:
        fixtures = fixtures[: int(args.limit)]

    if not fixtures:
        print("[]" if args.json else "No fixtures to probe.")
        return 0

    history = HistoryService()
    ctx_builder = ContextBuilder(history=history)
    models = default_models()
    agg = PredictionAggregatorImpl()

    results: list[Mapping[str, Any]] = []

    for f in fixtures:
        fx_id = f.get("fixture_id")
        league_id = f.get("league_id")
        season = f.get("season_year")
        home_id = f.get("home_id")
        away_id = f.get("away_id")

        if (
            not all(isinstance(v, int) for v in [fx_id, league_id, season])
            or home_id is None
            or away_id is None
        ):
            continue

        fx_id_i = cast(int, fx_id)
        league_i = cast(int, league_id)
        season_i = cast(int, season)
        home_i = cast(int, home_id)
        away_i = cast(int, away_id)

        ctx = ctx_builder.build_from_meta(
            fixture_id=fx_id_i,
            league_id=league_i,
            season=season_i,
            home_team_id=home_i,
            away_team_id=away_i,
        )

        match = Match(
            fixture_id=FixtureId(fx_id_i),
            league_id=LeagueId(league_i),
            season=season_i,
            kickoff_utc=datetime.now(timezone.utc),
            home_name=str(home_i),
            away_name=str(away_i),
            status=MatchStatus.SCHEDULED,
        )

        preds = agg.run_all(models, match, ctx)
        for p in preds:
            results.append(_serialize_prediction(p))

    if args.json:
        print(json.dumps(results, ensure_ascii=False))
    else:
        for r in results:
            if r.get("probs"):
                pr = r["probs"]
                print(
                    f"fixture={r['fixture_id']} model={r['model']} v{r['version']} "
                    f"home={pr['home']:.3f} draw={pr['draw']:.3f} away={pr['away']:.3f}"
                )
            else:
                print(
                    f"fixture={r['fixture_id']} model={r['model']} SKIPPED: {r.get('skip_reason','')}"
                )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
