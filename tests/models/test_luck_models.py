from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.domain.entities.match import Match
from src.domain.interfaces.context import ModelContext
from src.domain.value_objects.enums import MatchStatus, PredictionStatus
from src.domain.value_objects.ids import FixtureId, LeagueId, TeamId
from src.models.balance_luck import BalanceLuckHighModel, BalanceLuckLowModel
from src.models.veto_luck import VetoLuckHighModel, VetoLuckLowModel


def _make_match_ctx() -> tuple[Match, ModelContext]:
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
        home_team_id=TeamId(1),
        away_team_id=TeamId(2),
    )
    return match, ctx


class _DummyHistory:
    def __init__(self, home_rows: list[dict], away_rows: list[dict]):
        self.home_rows = home_rows
        self.away_rows = away_rows

    def get_recent_team_stats(
        self, team_id: int, league_id: int, season: int, last: int, *, only_finished: bool = True
    ) -> list[dict]:
        return self.home_rows if int(team_id) == 1 else self.away_rows


GOOD_HOME = [
    {"goals_for": 2, "goals_against": 0, "xg_for": 1.4, "xg_against": 0.6},
    {"goals_for": 1, "goals_against": 0, "xg_for": 1.1, "xg_against": 0.5},
]
GOOD_AWAY = [
    {"goals_for": 0, "goals_against": 2, "xg_for": 0.5, "xg_against": 1.5},
    {"goals_for": 0, "goals_against": 1, "xg_for": 0.6, "xg_against": 1.2},
]
MISSING_XG = [{"goals_for": 1, "goals_against": 1, "xg_for": None, "xg_against": None}]


@pytest.mark.parametrize("model_cls", [BalanceLuckLowModel, BalanceLuckHighModel])
def test_balance_luck_ok(model_cls: type) -> None:
    match, ctx = _make_match_ctx()
    history = _DummyHistory(GOOD_HOME, GOOD_AWAY)
    model = model_cls(history=history)
    pred = model.predict(match, ctx)
    assert pred.status == PredictionStatus.OK
    assert pred.probs is not None
    # Home stronger -> home prob should be highest
    assert pred.probs.home > pred.probs.away


def test_balance_luck_missing_xg_skips() -> None:
    match, ctx = _make_match_ctx()
    history = _DummyHistory(MISSING_XG, GOOD_AWAY)
    pred = BalanceLuckLowModel(history=history).predict(match, ctx)
    assert pred.status == PredictionStatus.SKIPPED
    assert pred.probs is None
    assert "xg" in (pred.skip_reason or "").lower()


@pytest.mark.parametrize("model_cls", [VetoLuckLowModel, VetoLuckHighModel])
def test_veto_luck_ok(model_cls: type) -> None:
    match, ctx = _make_match_ctx()
    history = _DummyHistory(GOOD_HOME, GOOD_AWAY)
    model = model_cls(history=history)
    pred = model.predict(match, ctx)
    assert pred.status == PredictionStatus.OK
    assert pred.probs is not None
    assert pred.probs.home > pred.probs.away


def test_veto_luck_missing_xg_skips() -> None:
    match, ctx = _make_match_ctx()
    history = _DummyHistory(GOOD_HOME, MISSING_XG)
    pred = VetoLuckLowModel(history=history).predict(match, ctx)
    assert pred.status == PredictionStatus.SKIPPED
    assert pred.probs is None
    assert "xg" in (pred.skip_reason or "").lower()
