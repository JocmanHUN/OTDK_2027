from __future__ import annotations

from datetime import datetime, timezone

from src.domain.entities.match import Match
from src.domain.interfaces.context import ModelContext
from src.domain.value_objects.enums import MatchStatus
from src.domain.value_objects.ids import FixtureId, LeagueId, TeamId
from src.models.monte_carlo import MonteCarloModel


def _mk_match_ctx(mu_h: float | None, mu_a: float | None) -> tuple[Match, ModelContext]:
    match = Match(
        fixture_id=FixtureId(2),
        league_id=LeagueId(39),
        season=2025,
        kickoff_utc=datetime.now(timezone.utc),
        home_name="H",
        away_name="A",
        status=MatchStatus.SCHEDULED,
    )
    ctx = ModelContext(
        fixture_id=FixtureId(2),
        league_id=LeagueId(39),
        season=2025,
        home_team_id=TeamId(1),
        away_team_id=TeamId(2),
        home_goal_rate=mu_h,
        away_goal_rate=mu_a,
    )
    return match, ctx


def test_monte_carlo_uniform_when_invalid_sims() -> None:
    model = MonteCarloModel(n_sims=0)
    match, ctx = _mk_match_ctx(1.0, 1.0)
    pred = model.predict(match, ctx)
    assert pred.probs is not None
    assert abs(pred.probs.home - 1 / 3) < 1e-12
    assert abs(pred.probs.draw - 1 / 3) < 1e-12
    assert abs(pred.probs.away - 1 / 3) < 1e-12


def test_monte_carlo_zero_lambdas_all_draws() -> None:
    model = MonteCarloModel(n_sims=1000, random_seed=123)
    match, ctx = _mk_match_ctx(0.0, 0.0)
    pred = model.predict(match, ctx)
    assert pred.probs is not None
    assert pred.probs.draw > 0.999
