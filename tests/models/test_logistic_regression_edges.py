from __future__ import annotations

from datetime import datetime, timezone

from src.domain.entities.match import Match
from src.domain.interfaces.context import ModelContext
from src.domain.value_objects.enums import MatchStatus
from src.domain.value_objects.ids import FixtureId, LeagueId, TeamId
from src.models.logistic_regression import LogisticRegressionModel


def _mk_match_ctx(features: dict[str, float] | None) -> tuple[Match, ModelContext]:
    match = Match(
        fixture_id=FixtureId(4),
        league_id=LeagueId(39),
        season=2025,
        kickoff_utc=datetime.now(timezone.utc),
        home_name="H",
        away_name="A",
        status=MatchStatus.SCHEDULED,
    )
    ctx = ModelContext(
        fixture_id=FixtureId(4),
        league_id=LeagueId(39),
        season=2025,
        home_team_id=TeamId(1),
        away_team_id=TeamId(2),
        features=features,
    )
    return match, ctx


def test_logreg_skips_without_features() -> None:
    match, ctx = _mk_match_ctx(None)
    model = LogisticRegressionModel()
    pred = model.predict(match, ctx)
    assert pred.status.name == "SKIPPED"


def test_logreg_handles_extreme_scores_and_draw_bounds() -> None:
    # Large positive score
    feats = {"diff_points_per_game": 1e6}
    match, ctx = _mk_match_ctx(feats)
    model = LogisticRegressionModel(base_draw=-1.0, draw_sensitivity=10.0)
    pred = model.predict(match, ctx)
    assert pred.probs is not None
    s = pred.probs.home + pred.probs.draw + pred.probs.away
    assert abs(s - 1.0) < 1e-12
    # Draw should be within [0, 0.6] due to clamp
    assert 0.0 <= pred.probs.draw <= 0.6
