# mypy: ignore-errors

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np

from src.domain.entities.match import Match
from src.domain.interfaces.context import ModelContext
from src.domain.value_objects.enums import MatchStatus, PredictionStatus
from src.domain.value_objects.ids import FixtureId, LeagueId, TeamId
from src.models.balance import BalanceModel
from src.models.balance_blend import BalanceBlendHighModel, BalanceBlendLowModel
from src.models.balance_luck import BalanceLuckLowModel
from src.models.balance_shift import BalanceShiftModel
from src.models.elo import EloModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.monte_carlo import MonteCarloModel
from src.models.poisson import PoissonModel
from src.models.veto import VetoModel
from src.models.veto_blend import VetoBlendHighModel, VetoBlendLowModel
from src.models.veto_luck import VetoLuckLowModel
from src.models.veto_shift import VetoShiftModel


def _match_ctx(with_teams: bool = True) -> tuple[Match, ModelContext]:
    match = Match(
        fixture_id=FixtureId(1),
        league_id=LeagueId(10),
        season=2024,
        kickoff_utc=datetime.now(timezone.utc),
        home_name="H",
        away_name="A",
        status=MatchStatus.SCHEDULED,
    )
    ctx = ModelContext(
        fixture_id=FixtureId(1),
        league_id=LeagueId(10),
        season=2024,
        home_team_id=TeamId(1) if with_teams else None,
        away_team_id=TeamId(2) if with_teams else None,
    )
    return match, ctx


class _Hist:
    def __init__(self, home_rows: list[dict], away_rows: list[dict]):
        self.home = home_rows
        self.away = away_rows

    def get_recent_team_stats(
        self, team_id: int, league_id: int, season: int, last: int, *, only_finished: bool = True
    ) -> list[dict]:
        return self.home if int(team_id) == 1 else self.away


GOOD_HOME = [
    {"goals_for": 2, "goals_against": 0, "xg_for": 1.5, "xg_against": 0.6},
    {"goals_for": 1, "goals_against": 0, "xg_for": 1.2, "xg_against": 0.5},
]
GOOD_AWAY = [
    {"goals_for": 0, "goals_against": 2, "xg_for": 0.4, "xg_against": 1.6},
    {"goals_for": 0, "goals_against": 1, "xg_for": 0.5, "xg_against": 1.3},
]
MISSING_XG = [{"goals_for": 1, "goals_against": 1, "xg_for": None, "xg_against": None}]
EMPTY = []


def test_balance_model_paths() -> None:
    match, ctx = _match_ctx()
    # Missing history
    pred = BalanceModel(history=None).predict(match, ctx)
    assert pred.status == PredictionStatus.SKIPPED
    # Missing team ids
    match2, ctx_no = _match_ctx(with_teams=False)
    pred = BalanceModel(history=_Hist(GOOD_HOME, GOOD_AWAY)).predict(match2, ctx_no)
    assert pred.status == PredictionStatus.SKIPPED
    # Happy path
    pred = BalanceModel(history=_Hist(GOOD_HOME, GOOD_AWAY)).predict(match, ctx)
    assert pred.status == PredictionStatus.OK and pred.probs is not None


def test_balance_blend_skip_and_ok() -> None:
    match, ctx = _match_ctx()
    h = _Hist(MISSING_XG, MISSING_XG)
    pred = BalanceBlendLowModel(history=h).predict(match, ctx)
    assert pred.status == PredictionStatus.SKIPPED and pred.probs is None
    pred_ok = BalanceBlendHighModel(history=_Hist(GOOD_HOME, GOOD_AWAY)).predict(match, ctx)
    assert pred_ok.status == PredictionStatus.OK and pred_ok.probs is not None


def test_balance_shift_paths() -> None:
    match, ctx = _match_ctx()
    hist = _Hist(EMPTY, GOOD_AWAY)
    pred = BalanceShiftModel(history=hist).predict(match, ctx)
    assert pred.status == PredictionStatus.SKIPPED
    hist_xg_missing = _Hist(MISSING_XG, GOOD_AWAY)
    pred = BalanceShiftModel(history=hist_xg_missing).predict(match, ctx)
    assert pred.status == PredictionStatus.SKIPPED
    pred_ok = BalanceShiftModel(history=_Hist(GOOD_HOME, GOOD_AWAY)).predict(match, ctx)
    assert pred_ok.status == PredictionStatus.OK and pred_ok.probs is not None


def test_balance_luck_missing_history_and_ids() -> None:
    match, ctx = _match_ctx()
    pred = BalanceLuckLowModel(history=None).predict(match, ctx)
    assert pred.status == PredictionStatus.SKIPPED
    match2, ctx_no = _match_ctx(with_teams=False)
    pred = BalanceLuckLowModel(history=_Hist(GOOD_HOME, GOOD_AWAY)).predict(match2, ctx_no)
    assert pred.status == PredictionStatus.SKIPPED


def test_veto_model_paths() -> None:
    match, ctx = _match_ctx()
    pred = VetoModel(history=None).predict(match, ctx)
    assert pred.status == PredictionStatus.SKIPPED
    match2, ctx_no = _match_ctx(with_teams=False)
    pred = VetoModel(history=_Hist(GOOD_HOME, GOOD_AWAY)).predict(match2, ctx_no)
    assert pred.status == PredictionStatus.SKIPPED
    pred_ok = VetoModel(history=_Hist(GOOD_HOME, GOOD_AWAY)).predict(match, ctx)
    assert pred_ok.status == PredictionStatus.OK and pred_ok.probs is not None


def test_veto_blend_paths() -> None:
    match, ctx = _match_ctx()
    pred = VetoBlendLowModel(history=None).predict(match, ctx)
    assert pred.status == PredictionStatus.SKIPPED
    match2, ctx_no = _match_ctx(with_teams=False)
    pred = VetoBlendLowModel(history=_Hist(GOOD_HOME, GOOD_AWAY)).predict(match2, ctx_no)
    assert pred.status == PredictionStatus.SKIPPED
    pred = VetoBlendLowModel(history=_Hist(MISSING_XG, GOOD_AWAY)).predict(match, ctx)
    assert pred.status == PredictionStatus.SKIPPED
    pred_ok = VetoBlendHighModel(history=_Hist(GOOD_HOME, GOOD_AWAY)).predict(match, ctx)
    assert pred_ok.status == PredictionStatus.OK and pred_ok.probs is not None


def test_veto_shift_paths() -> None:
    match, ctx = _match_ctx()
    pred = VetoShiftModel(history=None).predict(match, ctx)
    assert pred.status == PredictionStatus.SKIPPED
    match2, ctx_no = _match_ctx(with_teams=False)
    pred = VetoShiftModel(history=_Hist(GOOD_HOME, GOOD_AWAY)).predict(match2, ctx_no)
    assert pred.status == PredictionStatus.SKIPPED
    pred = VetoShiftModel(history=_Hist(EMPTY, GOOD_AWAY)).predict(match, ctx)
    assert pred.status == PredictionStatus.SKIPPED
    pred = VetoShiftModel(history=_Hist(MISSING_XG, GOOD_AWAY)).predict(match, ctx)
    assert pred.status == PredictionStatus.SKIPPED
    pred_ok = VetoShiftModel(history=_Hist(GOOD_HOME, GOOD_AWAY)).predict(match, ctx)
    assert pred_ok.status == PredictionStatus.OK and pred_ok.probs is not None


def test_veto_luck_missing_history_and_ids() -> None:
    match, ctx = _match_ctx()
    pred = VetoLuckLowModel(history=None).predict(match, ctx)
    assert pred.status == PredictionStatus.SKIPPED
    match2, ctx_no = _match_ctx(with_teams=False)
    pred = VetoLuckLowModel(history=_Hist(GOOD_HOME, GOOD_AWAY)).predict(match2, ctx_no)
    assert pred.status == PredictionStatus.SKIPPED


def test_logistic_regression_paths() -> None:
    match, ctx = _match_ctx()
    ctx_no_feats = ctx.model_copy(update={"features": None})
    pred = LogisticRegressionModel().predict(match, ctx_no_feats)
    assert pred.status == PredictionStatus.SKIPPED

    feats: Mapping[str, Any] = {
        "diff_goals_for_avg": 1.0,
        "diff_goals_against_avg": -0.5,
        "diff_points_per_game": 0.7,
        "diff_shots on target": 0.2,
        "diff_extra": 0.5,  # triggers generic diff_ branch
    }
    ctx_feats = ctx.model_copy(update={"features": feats})
    pred_ok = LogisticRegressionModel(base_draw=0.3, draw_sensitivity=0.4).predict(match, ctx_feats)
    assert pred_ok.status == PredictionStatus.OK and pred_ok.probs is not None


def test_poisson_paths() -> None:
    match, ctx = _match_ctx()
    ctx_missing = ctx.model_copy(update={"home_goal_rate": None, "away_goal_rate": None})
    pred = PoissonModel(tol=1e-6, max_goals=5).predict(match, ctx_missing)
    assert pred.status == PredictionStatus.SKIPPED

    ctx_rates = ctx.model_copy(update={"home_goal_rate": 1.2, "away_goal_rate": 0.8})
    pred_ok = PoissonModel(tol=1e-6, max_goals=3).predict(match, ctx_rates)
    assert pred_ok.status == PredictionStatus.OK and pred_ok.probs is not None


def test_monte_carlo_paths(monkeypatch) -> None:
    match, ctx = _match_ctx()
    ctx_missing = ctx.model_copy(update={"home_goal_rate": None, "away_goal_rate": None})
    pred = MonteCarloModel(n_sims=10).predict(match, ctx_missing)
    assert pred.status == PredictionStatus.SKIPPED

    # sims <=0 branch
    ctx_rates = ctx.model_copy(update={"home_goal_rate": 1.0, "away_goal_rate": 1.0})
    pred_uniform = MonteCarloModel(n_sims=0).predict(match, ctx_rates)
    assert pred_uniform.status == PredictionStatus.OK and pred_uniform.probs is not None

    # deterministic path
    pred_ok = MonteCarloModel(n_sims=1000, random_seed=42).predict(match, ctx_rates)
    assert pred_ok.status == PredictionStatus.OK and pred_ok.probs is not None
    assert np.isclose(pred_ok.probs.home + pred_ok.probs.draw + pred_ok.probs.away, 1.0, atol=1e-6)


def test_elo_paths(monkeypatch) -> None:
    match, ctx = _match_ctx()
    # Missing team IDs path
    match2, ctx_no = _match_ctx(with_teams=False)
    pred = EloModel(elo_service=object()).predict(match2, ctx_no)
    assert pred.status == PredictionStatus.SKIPPED

    # Use context elo directly
    ctx_with_elos = ctx.model_copy(update={"elo_home": 1600.0, "elo_away": 1500.0})
    pred_ok = EloModel(draw_param=0.2).predict(match, ctx_with_elos)
    assert pred_ok.status == PredictionStatus.OK and pred_ok.probs is not None

    # League ratings empty -> uniform
    class DummySvc:
        def get_league_ratings(self, league_id: int, season: int) -> dict[int, float]:
            return {}

        def get_team_rating(self, league_id: int, season: int, team_id: int) -> float:
            return 1500.0

    pred_uniform = EloModel(elo_service=DummySvc(), draw_param=-1.0).predict(match, ctx)
    assert pred_uniform.status == PredictionStatus.OK and pred_uniform.probs is not None
