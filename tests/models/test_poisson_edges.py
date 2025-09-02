from __future__ import annotations

from datetime import datetime, timezone

from src.domain.entities.match import Match
from src.domain.interfaces.context import ModelContext
from src.domain.value_objects.enums import MatchStatus
from src.domain.value_objects.ids import FixtureId, LeagueId, TeamId
from src.models.poisson import PoissonModel


def _mk_match_ctx(mu_h: float | None, mu_a: float | None) -> tuple[Match, ModelContext]:
    match = Match(
        fixture_id=FixtureId(1),
        league_id=LeagueId(39),
        season=2025,
        kickoff_utc=datetime.now(timezone.utc),
        home_name="H",
        away_name="A",
        status=MatchStatus.SCHEDULED,
    )
    ctx = ModelContext(
        fixture_id=FixtureId(1),
        league_id=LeagueId(39),
        season=2025,
        home_team_id=TeamId(1),
        away_team_id=TeamId(2),
        home_goal_rate=mu_h,
        away_goal_rate=mu_a,
    )
    return match, ctx


def test_poisson_negative_rates_behave_as_zero() -> None:
    m = PoissonModel(tol=0.0, max_goals=-5)
    match, ctx = _mk_match_ctx(-1.0, -2.0)
    pred = m.predict(match, ctx)
    assert pred.probs is not None
    # With both lambdas effectively zero, draw should be ~1.0
    assert pred.probs.draw > 0.9999


def test_poisson_zero_and_positive_rates_normalize() -> None:
    m = PoissonModel(tol=1e-9, max_goals=0)
    match, ctx = _mk_match_ctx(0.0, 1.2)
    pred = m.predict(match, ctx)
    assert pred.probs is not None
    s = pred.probs.home + pred.probs.draw + pred.probs.away
    assert abs(s - 1.0) < 1e-12
