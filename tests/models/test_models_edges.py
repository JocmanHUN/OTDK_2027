# mypy: ignore-errors

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.domain.entities.match import Match
from src.domain.interfaces.context import ModelContext
from src.domain.value_objects.enums import MatchStatus, PredictionStatus
from src.domain.value_objects.ids import FixtureId, LeagueId, TeamId
from src.models import balance as balance_mod
from src.models import balance_blend as balance_blend_mod
from src.models import balance_luck as balance_luck_mod
from src.models import balance_shift as balance_shift_mod
from src.models import blend_models, default_models, luck_variants, shift_models
from src.models import veto as veto_mod
from src.models import veto_blend as veto_blend_mod
from src.models import veto_luck as veto_luck_mod
from src.models import veto_shift as veto_shift_mod
from src.models.balance_blend import BalanceBlendHighModel, BalanceBlendLowModel
from src.models.logistic_regression import LogisticRegressionModel, _sigmoid
from src.models.poisson import PoissonModel, _poisson_pmf_vector
from src.models.veto_blend import VetoBlendHighModel, VetoBlendLowModel


def _mk_match_ctx(with_ids: bool = True) -> tuple[Match, ModelContext]:
    m = Match(
        fixture_id=FixtureId(1),
        league_id=LeagueId(2),
        season=2024,
        kickoff_utc=datetime.now(timezone.utc),
        home_name="H",
        away_name="A",
        status=MatchStatus.SCHEDULED,
    )
    ctx = ModelContext(
        fixture_id=FixtureId(1),
        league_id=LeagueId(2),
        season=2024,
        home_team_id=TeamId(1) if with_ids else None,
        away_team_id=TeamId(3) if with_ids else None,
    )
    return m, ctx


def test_balance_blend_missing_ids_and_history() -> None:
    match, ctx_no = _mk_match_ctx(with_ids=False)
    pred = BalanceBlendLowModel(history=None).predict(match, ctx_no)
    assert pred.status == PredictionStatus.SKIPPED

    # Missing history branch
    match2, ctx = _mk_match_ctx()
    pred = BalanceBlendHighModel(history=None).predict(match2, ctx)
    assert pred.status == PredictionStatus.SKIPPED


def test_veto_blend_missing_ids_history() -> None:
    match, ctx_no = _mk_match_ctx(with_ids=False)
    pred = VetoBlendLowModel(history=None).predict(match, ctx_no)
    assert pred.status == PredictionStatus.SKIPPED

    match2, ctx = _mk_match_ctx()
    pred = VetoBlendHighModel(history=None).predict(match2, ctx)
    assert pred.status == PredictionStatus.SKIPPED


def test_poisson_zero_mu_branch() -> None:
    pmf = _poisson_pmf_vector(0.0)
    assert pmf == [1.0]
    match, ctx = _mk_match_ctx()
    ctx_rates = ctx.model_copy(update={"home_goal_rate": 0.0, "away_goal_rate": 0.0})
    pred = PoissonModel().predict(match, ctx_rates)
    assert pred.status == PredictionStatus.OK and pred.probs is not None


def test_logistic_sigmoid_edges() -> None:
    assert _sigmoid(1000) == pytest.approx(1.0)
    assert _sigmoid(-1000) == pytest.approx(0.0)
    match, ctx = _mk_match_ctx()
    feats = {"diff_goals_for_avg": -10.0}
    pred = LogisticRegressionModel(base_draw=0.9, draw_sensitivity=0.0).predict(
        match, ctx.model_copy(update={"features": feats})
    )
    assert pred.status == PredictionStatus.OK and pred.probs is not None


def test_models_init_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    # Use a lightweight HistoryService stub to avoid hitting real services
    import src.models as models_mod

    class DummyHistory:
        def __init__(self) -> None:
            pass

    monkeypatch.setattr(models_mod, "HistoryService", lambda: DummyHistory())
    # Should instantiate without errors
    all_models = default_models()
    assert any(getattr(m, "name", None) for m in all_models)

    blends = blend_models(mix_weight=0.2)
    assert len(blends) == 2

    shifts = shift_models(margin_weight=0.5, margin_clip=1.0)
    assert len(shifts) == 2

    lucks = luck_variants()
    assert len(lucks) == 6


def test_balance_distribution_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_max = builtins.max

    def fake_max(a, b):
        return -1.0 if (a == 0.0 and b <= 1.0) else real_max(a, b)

    monkeypatch.setattr("builtins.max", fake_max)
    rows = [{"goals_for": 1, "goals_against": 0}, {"goals_for": 0, "goals_against": 1}]
    assert balance_mod._form_distribution(rows, 0.5) == (1 / 3, 1 / 3, 1 / 3)
    monkeypatch.setattr("builtins.max", real_max)

    # df == 0 branch (no fake max)
    rows_single = [{"goals_for": 1, "goals_against": 0}]
    assert balance_mod._form_distribution(rows_single, 0.0)[0] > 0

    # s <= 0 branch in predict
    match, ctx = _mk_match_ctx()
    monkeypatch.setattr(balance_mod, "_form_distribution", lambda *_, **__: (0.0, 0.0, 0.0))

    # Use BalanceModel with patched helper and dummy history
    class _H:
        def get_recent_team_stats(self, *args, **kwargs):
            return [{}]

    bm = balance_mod.BalanceModel(history=_H())
    pred_bm = bm.predict(match, ctx)
    assert pred_bm.status == PredictionStatus.OK and pred_bm.probs is not None


def test_balance_blend_helper_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_max = builtins.max

    def fake_max(a, b):
        return -1.0 if (a == 0.0 and b <= 1.0) else real_max(a, b)

    monkeypatch.setattr("builtins.max", fake_max)
    rows = [{"goals_for": 1, "goals_against": 0, "xg_for": 1.0, "xg_against": 0.5}] * 2
    assert balance_blend_mod._form_distribution(rows, 0.5) == (1 / 3, 1 / 3, 1 / 3)
    assert balance_blend_mod._form_xg_distribution(rows, 0.5, 0.1) == (1 / 3, 1 / 3, 1 / 3)
    monkeypatch.setattr("builtins.max", real_max)

    match, ctx = _mk_match_ctx()
    # Force s<=0 in predict by patching helpers
    monkeypatch.setattr(balance_blend_mod, "_form_distribution", lambda *_, **__: (0.0, 0.0, 0.0))

    class _H:
        def get_recent_team_stats(self, *args, **kwargs):
            return [{"goals_for": 0, "goals_against": 0, "xg_for": 0.0, "xg_against": 0.0}]

    pred = balance_blend_mod.BalanceBlendLowModel(history=_H()).predict(match, ctx)
    assert pred.status == PredictionStatus.OK and pred.probs is not None


def test_balance_luck_helper_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_max = builtins.max

    def fake_max(a, b):
        return -1.0 if (a == 0.0 and b <= 1.0) else real_max(a, b)

    monkeypatch.setattr("builtins.max", fake_max)
    rows = [{"goals_for": 1, "goals_against": 0, "xg_for": 0.5, "xg_against": 0.5}] * 2
    assert balance_luck_mod._form_distribution_with_luck(rows, 0.5, 0.1, 1.0) == (
        1 / 3,
        1 / 3,
        1 / 3,
    )
    monkeypatch.setattr("builtins.max", real_max)


def test_balance_shift_helper_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_max = builtins.max

    def fake_max(a, b):
        return -1.0 if (a == 0.0 and b <= 1.0) else real_max(a, b)

    monkeypatch.setattr("builtins.max", fake_max)
    rows = [{"goals_for": 1, "goals_against": 0, "xg_for": 1.0, "xg_against": 0.5}] * 2
    assert balance_shift_mod._form_distribution(rows, 0.5) == (1 / 3, 1 / 3, 1 / 3)
    assert balance_shift_mod._xg_margin(rows, 0.5) == 0.0
    monkeypatch.setattr("builtins.max", real_max)
    # softmax_shift s<=0
    assert balance_shift_mod._softmax_shift(0.0, 0.0, 0.0, 0.0) == (1 / 3, 1 / 3, 1 / 3)


def test_veto_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_max = builtins.max

    def fake_max(a, b):
        return -1.0 if (a == 0.0 and b <= 1.0) else real_max(a, b)

    monkeypatch.setattr("builtins.max", fake_max)
    rows = [{"goals_for": 1, "goals_against": 0}] * 2
    assert veto_mod._form_distribution(rows, 0.5) == (1 / 3, 1 / 3, 1 / 3)
    monkeypatch.setattr("builtins.max", real_max)


def test_veto_blend_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_max = builtins.max

    def fake_max(a, b):
        return -1.0 if (a == 0.0 and b <= 1.0) else real_max(a, b)

    monkeypatch.setattr("builtins.max", fake_max)
    rows = [{"goals_for": 1, "goals_against": 0, "xg_for": 1.0, "xg_against": 0.5}] * 2
    assert veto_blend_mod._form_distribution(rows, 0.5) == (1 / 3, 1 / 3, 1 / 3)
    assert veto_blend_mod._form_xg_distribution(rows, 0.5, 0.1) == (1 / 3, 1 / 3, 1 / 3)
    monkeypatch.setattr("builtins.max", real_max)


def test_veto_luck_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_max = builtins.max

    def fake_max(a, b):
        return -1.0 if (a == 0.0 and b <= 1.0) else real_max(a, b)

    monkeypatch.setattr("builtins.max", fake_max)
    rows = [{"goals_for": 1, "goals_against": 0, "xg_for": 0.5, "xg_against": 0.5}] * 2
    assert veto_luck_mod._form_distribution_with_luck(rows, 0.5, 0.1, 1.0) == (
        1 / 3,
        1 / 3,
        1 / 3,
    )
    monkeypatch.setattr("builtins.max", real_max)


def test_veto_shift_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_max = builtins.max

    def fake_max(a, b):
        return -1.0 if (a == 0.0 and b <= 1.0) else real_max(a, b)

    monkeypatch.setattr("builtins.max", fake_max)
    rows = [{"goals_for": 1, "goals_against": 0, "xg_for": 1.0, "xg_against": 0.5}] * 2
    assert veto_shift_mod._form_distribution(rows, 0.5) == (1 / 3, 1 / 3, 1 / 3)
    monkeypatch.setattr("builtins.max", real_max)
    import math as _math

    monkeypatch.setattr(veto_shift_mod.math, "exp", lambda x: 0.0)
    assert veto_shift_mod._softmax_shift(0.1, 0.1, 0.1, 0.0) == (1 / 3, 1 / 3, 1 / 3)
    monkeypatch.setattr(veto_shift_mod.math, "exp", _math.exp)


def test_logistic_overflow_and_bad_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force OverflowError path
    def _raise_overflow(x):
        raise OverflowError()

    monkeypatch.setattr("src.models.logistic_regression.exp", _raise_overflow)
    assert _sigmoid(5.0) in (0.0, 1.0)
    # Bad feature value triggers except branch and foul keyword triggers negative weight
    match, ctx = _mk_match_ctx()
    feats = {"diff_goals_for_avg": 0.0, "diff_extra_bad": "bad", "diff_red cards": 1.0}
    pred = LogisticRegressionModel().predict(match, ctx.model_copy(update={"features": feats}))
    assert pred.status == PredictionStatus.OK and pred.probs is not None

    # Explicit negative-weight branch
    feats_neg = {"diff_red cards": 1.0, "diff_foul_play": 2.0}
    pred2 = LogisticRegressionModel().predict(match, ctx.model_copy(update={"features": feats_neg}))
    assert pred2.status == PredictionStatus.OK and pred2.probs is not None


def test_poisson_zero_total_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    match, ctx = _mk_match_ctx()
    ctx_rates = ctx.model_copy(update={"home_goal_rate": 1.0, "away_goal_rate": 1.0})
    monkeypatch.setattr("src.models.poisson._outer_sum", lambda *args, **kwargs: (0.0, 0.0, 0.0))
    pred = PoissonModel().predict(match, ctx_rates)
    assert pred.status == PredictionStatus.OK and pred.probs is not None


def test_remaining_softmax_and_negative_sums(monkeypatch: pytest.MonkeyPatch) -> None:
    # Balance shift softmax fallback via patched exp -> s == 0
    import math as _math

    monkeypatch.setattr(balance_shift_mod.math, "exp", lambda x: 0.0)
    assert balance_shift_mod._softmax_shift(0.1, 0.1, 0.1, 0.0) == (1 / 3, 1 / 3, 1 / 3)
    monkeypatch.setattr(balance_shift_mod.math, "exp", _math.exp)

    match, ctx = _mk_match_ctx()

    # Veto blend s<=0
    def _fake_form(rows, decay, *args, **kwargs):
        val = rows[0]["val"]
        return (val, val, val)

    monkeypatch.setattr(veto_blend_mod, "_form_distribution", _fake_form)
    monkeypatch.setattr(veto_blend_mod, "_form_xg_distribution", _fake_form)

    class HVB2:
        def get_recent_team_stats(self, team_id: int, *args, **kwargs):
            if int(team_id) == 1:
                return [
                    {"val": 1, "xg_for": 0.0, "xg_against": 0.0, "goals_for": 0, "goals_against": 0}
                ]
            return [
                {"val": -1, "xg_for": 0.0, "xg_against": 0.0, "goals_for": 0, "goals_against": 0}
            ]

    pred = veto_blend_mod.VetoBlendLowModel(history=HVB2(), mul_weight=1.0, mix_weight=0.5).predict(
        match, ctx
    )
    assert pred.probs is not None and pred.probs.home == pytest.approx(1 / 3)

    # Veto luck s<=0
    def _fake_luck(rows, decay, *args, **kwargs):
        val = rows[0]["val"]
        return (val, val, val)

    monkeypatch.setattr(veto_luck_mod, "_form_distribution_with_luck", _fake_luck)

    class HVL2:
        def get_recent_team_stats(self, team_id: int, *args, **kwargs):
            if int(team_id) == 1:
                return [
                    {"val": 1, "xg_for": 0.0, "xg_against": 0.0, "goals_for": 0, "goals_against": 0}
                ]
            return [
                {"val": -1, "xg_for": 0.0, "xg_against": 0.0, "goals_for": 0, "goals_against": 0}
            ]

    pred = veto_luck_mod.VetoLuckLowModel(history=HVL2()).predict(match, ctx)
    assert pred.probs is not None and pred.probs.home == pytest.approx(1 / 3)

    # Veto shift xg_margin w_sum<=0 and softmax fallback
    import builtins

    real_max = builtins.max

    def fake_max(a, b):
        return -1.0 if (a == 0.0 and b <= 1.0) else real_max(a, b)

    monkeypatch.setattr("builtins.max", fake_max)
    rows = [{"xg_for": 1.0, "xg_against": 0.5}] * 2
    assert veto_shift_mod._xg_margin(rows, 0.5) == 0.0
    monkeypatch.setattr("builtins.max", real_max)
    assert veto_shift_mod._softmax_shift(0.0, 0.0, 0.0, 0.0) == (1 / 3, 1 / 3, 1 / 3)


def test_elo_extra_branches() -> None:
    from src.models.elo import _davidson_probs

    # denom <= 0
    assert _davidson_probs(0.0, -2.0) == (1 / 3, 1 / 3, 1 / 3)

    match, ctx = _mk_match_ctx()

    class DummySvc:
        def get_league_ratings(self, league_id: int, season: int):
            raise RuntimeError("boom")

        def get_team_rating(self, league_id: int, season: int, team_id: int) -> float:
            return 1500.0

    from src.models.elo import EloModel

    em = EloModel(elo_service=DummySvc(), draw_param=None)
    pred = em.predict(match, ctx)
    assert pred.status == PredictionStatus.OK and pred.probs is not None


def test_helper_empties_and_negative_sums(monkeypatch: pytest.MonkeyPatch) -> None:
    match, ctx = _mk_match_ctx()
    # balance empty rows and df==0
    assert balance_mod._form_distribution([], 0.5) == (1 / 3, 1 / 3, 1 / 3)
    balance_mod._form_distribution([{"goals_for": 0, "goals_against": 0}], 0.0)

    # force s<=0
    class H:
        def get_recent_team_stats(self, *args, **kwargs):
            return [{"goals_for": 0, "goals_against": 0}]

    monkeypatch.setattr(balance_mod, "_form_distribution", lambda *_, **__: (-10.0, -10.0, -10.0))
    pred = balance_mod.BalanceModel(history=H()).predict(match, ctx)
    assert pred.probs is not None and pred.probs.home == pytest.approx(1 / 3)

    # balance_blend helpers
    assert balance_blend_mod._form_distribution([], 0.5) == (1 / 3, 1 / 3, 1 / 3)
    balance_blend_mod._form_distribution([{"goals_for": 1, "goals_against": 1}], 0.0)
    assert balance_blend_mod._form_xg_distribution([], 0.5, 0.1) == (1 / 3, 1 / 3, 1 / 3)
    balance_blend_mod._form_xg_distribution([{"xg_for": 0.0, "xg_against": 0.0}], 0.0, 0.1)
    assert balance_blend_mod._rows_have_xg([]) is False
    monkeypatch.setattr(
        balance_blend_mod, "_form_distribution", lambda *_, **__: (-10.0, -10.0, -10.0)
    )
    monkeypatch.setattr(
        balance_blend_mod, "_form_xg_distribution", lambda *_, **__: (-10.0, -10.0, -10.0)
    )

    class HB:
        def get_recent_team_stats(self, *args, **kwargs):
            return [{"goals_for": 0, "goals_against": 0, "xg_for": 0.0, "xg_against": 0.0}]

    pred = balance_blend_mod.BalanceBlendLowModel(history=HB()).predict(match, ctx)
    assert pred.probs is not None and pred.probs.home == pytest.approx(1 / 3)

    # balance_luck helper empties
    assert balance_luck_mod._form_distribution_with_luck([], 0.5, 0.5, 0.5) == (
        1 / 3,
        1 / 3,
        1 / 3,
    )
    balance_luck_mod._form_distribution_with_luck(
        [{"goals_for": 1, "goals_against": 0, "xg_for": 0.0, "xg_against": 0.0}], 0.0, 0.5, 0.5
    )
    balance_luck_mod._form_distribution_with_luck(
        [{"goals_for": 1, "goals_against": 1, "xg_for": 0.0, "xg_against": 0.0}], 0.5, 0.5, 0.5
    )
    assert balance_luck_mod._rows_have_xg([]) is False

    class HL:
        def get_recent_team_stats(self, *args, **kwargs):
            return [{"goals_for": 0, "goals_against": 0, "xg_for": 0.0, "xg_against": 0.0}]

    monkeypatch.setattr(
        balance_luck_mod,
        "_form_distribution_with_luck",
        lambda *_, **__: (-10.0, -10.0, -10.0),
    )
    pred = balance_luck_mod.BalanceLuckLowModel(history=HL()).predict(match, ctx)
    assert pred.probs is not None and pred.probs.home == pytest.approx(1 / 3)

    # balance_shift helpers
    assert balance_shift_mod._form_distribution([], 0.5) == (1 / 3, 1 / 3, 1 / 3)
    balance_shift_mod._form_distribution([{"goals_for": 0, "goals_against": 0}], 0.0)
    assert balance_shift_mod._xg_margin([], 0.5) == 0.0
    balance_shift_mod._xg_margin([{"xg_for": 0.0, "xg_against": 0.0}], 0.0)
    assert balance_shift_mod._softmax_shift(0.0, 0.0, 0.0, 0.0) == (1 / 3, 1 / 3, 1 / 3)
    # missing ids/history skips
    bm_shift = balance_shift_mod.BalanceShiftModel(history=None)
    pred_skip = bm_shift.predict(match, ctx.model_copy(update={"home_team_id": None}))
    assert pred_skip.status == PredictionStatus.SKIPPED
    pred_skip2 = balance_shift_mod.BalanceShiftModel(history=None).predict(match, ctx)
    assert pred_skip2.status == PredictionStatus.SKIPPED

    # veto helpers
    assert veto_mod._form_distribution([], 0.5) == (1 / 3, 1 / 3, 1 / 3)
    veto_mod._form_distribution([{"goals_for": 1, "goals_against": 1}], 0.0)

    class HV:
        def get_recent_team_stats(self, *args, **kwargs):
            return []

    pred_v = veto_mod.VetoModel(history=HV()).predict(match, ctx)
    assert pred_v.status == PredictionStatus.OK and pred_v.probs is not None
    monkeypatch.setattr(veto_mod, "_form_distribution", lambda *_, **__: (-10.0, -10.0, -10.0))
    pred_v2 = veto_mod.VetoModel(history=HV()).predict(match, ctx)
    assert pred_v2.probs is not None and pred_v2.probs.home == pytest.approx(1 / 3)

    # veto_blend helpers
    assert veto_blend_mod._form_distribution([], 0.5) == (1 / 3, 1 / 3, 1 / 3)
    veto_blend_mod._form_distribution([{"goals_for": 1, "goals_against": 1}], 0.0)
    assert veto_blend_mod._form_xg_distribution([], 0.5, 0.1) == (1 / 3, 1 / 3, 1 / 3)
    veto_blend_mod._form_xg_distribution([{"xg_for": 0.0, "xg_against": 0.0}], 0.0, 0.1)
    monkeypatch.setattr(
        veto_blend_mod, "_form_distribution", lambda *_, **__: (-10.0, -10.0, -10.0)
    )
    monkeypatch.setattr(
        veto_blend_mod, "_form_xg_distribution", lambda *_, **__: (-10.0, -10.0, -10.0)
    )
    assert veto_blend_mod._rows_have_xg([]) is False

    class HVB:
        def get_recent_team_stats(self, *args, **kwargs):
            return [{"goals_for": 0, "goals_against": 0, "xg_for": 0.0, "xg_against": 0.0}]

    pred = veto_blend_mod.VetoBlendLowModel(history=HVB()).predict(match, ctx)
    assert pred.probs is not None and pred.probs.home == pytest.approx(1 / 3)

    # veto_luck helpers
    assert veto_luck_mod._form_distribution_with_luck([], 0.5, 0.5, 0.5) == (
        1 / 3,
        1 / 3,
        1 / 3,
    )
    veto_luck_mod._form_distribution_with_luck(
        [{"goals_for": 1, "goals_against": 0, "xg_for": 0.0, "xg_against": 0.0}], 0.0, 0.5, 0.5
    )
    veto_luck_mod._form_distribution_with_luck(
        [{"goals_for": 1, "goals_against": 1, "xg_for": 0.0, "xg_against": 0.0}], 0.5, 0.5, 0.5
    )
    assert veto_luck_mod._rows_have_xg([]) is False

    class HVL:
        def get_recent_team_stats(self, *args, **kwargs):
            return [{"goals_for": 0, "goals_against": 0, "xg_for": 0.0, "xg_against": 0.0}]

    monkeypatch.setattr(
        veto_luck_mod, "_form_distribution_with_luck", lambda *_, **__: (-10.0, -10.0, -10.0)
    )
    pred = veto_luck_mod.VetoLuckLowModel(history=HVL()).predict(match, ctx)
    assert pred.probs is not None and pred.probs.home == pytest.approx(1 / 3)

    # veto_shift helpers
    assert veto_shift_mod._form_distribution([], 0.5) == (1 / 3, 1 / 3, 1 / 3)
    veto_shift_mod._form_distribution([{"goals_for": 1, "goals_against": 1}], 0.0)
    assert veto_shift_mod._softmax_shift(0.0, 0.0, 0.0, 0.0) == (1 / 3, 1 / 3, 1 / 3)
    assert veto_shift_mod._xg_margin([], 0.5) == 0.0
    veto_shift_mod._xg_margin([{"xg_for": 0.0, "xg_against": 0.0}], 0.0)
