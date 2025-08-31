from __future__ import annotations

from datetime import datetime, timezone
from math import exp, sqrt

from src.domain.entities.match import Match
from src.domain.interfaces.context import ModelContext
from src.domain.value_objects.enums import MatchStatus
from src.domain.value_objects.ids import FixtureId, LeagueId, TeamId
from src.models.elo import EloModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.monte_carlo import MonteCarloModel
from src.models.poisson import PoissonModel


def _mk_match_ctx(
    *,
    league_id: int = 39,
    season: int = 2025,
    fixture_id: int = 100,
    home_id: int = 1,
    away_id: int = 2,
    mu_home: float | None = None,
    mu_away: float | None = None,
    elo_home: float | None = None,
    elo_away: float | None = None,
    features: dict[str, float] | None = None,
) -> tuple[Match, ModelContext]:
    match = Match(
        fixture_id=FixtureId(int(fixture_id)),
        league_id=LeagueId(int(league_id)),
        season=int(season),
        kickoff_utc=datetime.now(timezone.utc),
        home_name="H",
        away_name="A",
        status=MatchStatus.SCHEDULED,
    )
    ctx = ModelContext(
        fixture_id=FixtureId(int(fixture_id)),
        league_id=LeagueId(int(league_id)),
        season=int(season),
        home_team_id=TeamId(int(home_id)),
        away_team_id=TeamId(int(away_id)),
        home_goal_rate=mu_home,
        away_goal_rate=mu_away,
        elo_home=elo_home,
        elo_away=elo_away,
        home_advantage=80.0,
        features=features,
    )
    return match, ctx


def _poisson_pmf(mu: float, kmax: int = 12) -> list[float]:
    out = []
    for k in range(kmax + 1):
        # p(k) = e^{-mu} mu^k / k!
        # compute iteratively to avoid factorials
        if k == 0:
            out.append(exp(-mu))
        else:
            out.append(out[-1] * mu / k)
    # normalize in case of truncation
    s = sum(out)
    return [p / s for p in out]


def _outer_1x2(ph: list[float], pa: list[float]) -> tuple[float, float, float]:
    p_home = p_draw = p_away = 0.0
    for i, pi in enumerate(ph):
        for j, pj in enumerate(pa):
            pij = pi * pj
            if i > j:
                p_home += pij
            elif i == j:
                p_draw += pij
            else:
                p_away += pij
    s = p_home + p_draw + p_away
    return p_home / s, p_draw / s, p_away / s


def test_poisson_model_basic() -> None:
    match, ctx = _mk_match_ctx(mu_home=1.6, mu_away=1.0)
    m = PoissonModel()
    pred = m.predict(match, ctx)
    assert pred.probs is not None
    s = pred.probs.home + pred.probs.draw + pred.probs.away
    assert abs(s - 1.0) < 1e-9
    # sanity: home stronger than away
    assert pred.probs.home > pred.probs.away
    # compare to truncated exact
    ph = _poisson_pmf(1.6)
    pa = _poisson_pmf(1.0)
    eh, ed, ea = _outer_1x2(ph, pa)
    assert abs(pred.probs.home - eh) < 0.03
    assert abs(pred.probs.draw - ed) < 0.03
    assert abs(pred.probs.away - ea) < 0.03


def test_monte_carlo_matches_poisson_in_expectation() -> None:
    match, ctx = _mk_match_ctx(mu_home=1.4, mu_away=0.9)
    mc = MonteCarloModel(n_sims=20000, random_seed=123)
    pred = mc.predict(match, ctx)
    assert pred.probs is not None
    ph = _poisson_pmf(1.4)
    pa = _poisson_pmf(0.9)
    eh, ed, ea = _outer_1x2(ph, pa)
    # looser tolerance due to randomness
    assert abs(pred.probs.home - eh) < 0.05
    assert abs(pred.probs.draw - ed) < 0.05
    assert abs(pred.probs.away - ea) < 0.05


def test_elo_model_davidson_draw() -> None:
    # Premier League baseline H=80 from seed, elo_home higher
    match, ctx = _mk_match_ctx(league_id=39, elo_home=1550.0, elo_away=1500.0)
    elo = EloModel()
    pred = elo.predict(match, ctx)
    assert pred.probs is not None
    # Expected via Davidson with nu=0.28 and delta = 50 + H(80) = 130
    delta = 1550.0 - 1500.0 + 80.0
    gamma = 10.0 ** (delta / 400.0)
    root = sqrt(gamma)
    denom = gamma + 1.0 + 2.0 * 0.28 * root
    eh = gamma / denom
    ed = (2.0 * 0.28 * root) / denom
    ea = 1.0 / denom
    assert abs(pred.probs.home - eh) < 1e-6
    assert abs(pred.probs.draw - ed) < 1e-6
    assert abs(pred.probs.away - ea) < 1e-6


def test_logistic_regression_uses_feature_diffs() -> None:
    feats = {
        "diff_goals_for_avg": 0.6,
        "diff_goals_against_avg": -0.4,
        "diff_points_per_game": 0.8,
        "diff_shots on target": 2.0,
        "diff_ball possession": 10.0,
        "diff_corners": 1.0,
    }
    match, ctx = _mk_match_ctx(features=feats)
    lr = LogisticRegressionModel(base_draw=0.25, draw_sensitivity=0.3)
    pred = lr.predict(match, ctx)
    assert pred.probs is not None
    s = pred.probs.home + pred.probs.draw + pred.probs.away
    assert abs(s - 1.0) < 1e-9
    # With strong home-leaning features, expect home > away
    assert pred.probs.home > pred.probs.away
    # Draw should be in a plausible range
    assert 0.1 <= pred.probs.draw <= 0.6
