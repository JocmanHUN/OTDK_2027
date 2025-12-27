from __future__ import annotations

from datetime import datetime, timezone

from src.domain.entities.match import Match
from src.domain.interfaces.context import ModelContext
from src.domain.value_objects.enums import MatchStatus, PredictionStatus
from src.domain.value_objects.ids import FixtureId, LeagueId, TeamId
from src.models.balance_blend import BalanceBlendLowModel, BalanceBlendModel
from src.models.balance_shift import BalanceShiftModel
from src.models.veto_blend import VetoBlendLowModel, VetoBlendModel
from src.models.veto_shift import VetoShiftModel


def _make_match_and_ctx() -> tuple[Match, ModelContext]:
    match = Match(
        fixture_id=FixtureId(1),
        league_id=LeagueId(10),
        season=2024,
        kickoff_utc=datetime.now(timezone.utc),
        home_name="Home",
        away_name="Away",
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
        self._home_rows = home_rows
        self._away_rows = away_rows

    def get_recent_team_stats(
        self, team_id: int, league_id: int, season: int, last: int, *, only_finished: bool = True
    ) -> list[dict]:
        return self._home_rows if int(team_id) == 1 else self._away_rows


GOOD_HOME = [
    {"goals_for": 2, "goals_against": 0, "xg_for": 1.8, "xg_against": 0.7},
    {"goals_for": 1, "goals_against": 0, "xg_for": 1.2, "xg_against": 0.6},
]
GOOD_AWAY = [
    {"goals_for": 0, "goals_against": 2, "xg_for": 0.4, "xg_against": 1.6},
    {"goals_for": 0, "goals_against": 1, "xg_for": 0.5, "xg_against": 1.3},
]
MISSING_XG = [{"goals_for": 1, "goals_against": 1, "xg_for": None, "xg_against": None}]


def test_balance_blend_low_predicts_and_aliases() -> None:
    match, ctx = _make_match_and_ctx()
    history = _DummyHistory(GOOD_HOME, GOOD_AWAY)
    model = BalanceBlendLowModel(history=history, mix_weight=0.3, xg_threshold=0.1)
    pred = model.predict(match, ctx)

    assert pred.status == PredictionStatus.OK
    assert pred.probs is not None
    # Strong home signal: home probability should dominate
    assert pred.probs.home > 0.7 and pred.probs.home > pred.probs.away
    # Backward-compat alias still points to the Low variant
    assert BalanceBlendModel is BalanceBlendLowModel


def test_balance_blend_skips_without_xg() -> None:
    match, ctx = _make_match_and_ctx()
    history = _DummyHistory(MISSING_XG, MISSING_XG)
    pred = BalanceBlendLowModel(history=history).predict(match, ctx)

    assert pred.status == PredictionStatus.SKIPPED
    assert pred.probs is None
    assert "xg" in (pred.skip_reason or "").lower()


def test_veto_blend_low_predicts() -> None:
    match, ctx = _make_match_and_ctx()
    history = _DummyHistory(GOOD_HOME, GOOD_AWAY)
    model = VetoBlendLowModel(history=history, mix_weight=0.3, mul_weight=0.6, xg_threshold=0.1)
    pred = model.predict(match, ctx)

    assert pred.status == PredictionStatus.OK
    assert pred.probs is not None
    assert pred.probs.home > pred.probs.away
    # Legacy alias preserved
    assert VetoBlendModel is VetoBlendLowModel


def test_balance_shift_increases_home_prob_with_positive_margin() -> None:
    match, ctx = _make_match_and_ctx()
    history = _DummyHistory(GOOD_HOME, GOOD_AWAY)

    base = BalanceShiftModel(history=history, margin_weight=0.0, margin_clip=2.0).predict(
        match, ctx
    )
    shifted = BalanceShiftModel(history=history, margin_weight=1.0, margin_clip=2.0).predict(
        match, ctx
    )

    assert base.probs is not None and shifted.probs is not None
    # Positive home xG margin should increase the home probability relative to zero shift
    assert shifted.probs.home > base.probs.home


def test_veto_shift_skips_when_missing_xg() -> None:
    match, ctx = _make_match_and_ctx()
    history = _DummyHistory(MISSING_XG, GOOD_AWAY)
    pred = VetoShiftModel(history=history).predict(match, ctx)

    assert pred.status == PredictionStatus.SKIPPED
    assert pred.probs is None
    assert "xg" in (pred.skip_reason or "").lower()
