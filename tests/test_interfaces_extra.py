import importlib
import sys

from src.domain.interfaces.context import ModelContext
from src.domain.value_objects.enums import ModelName
from src.domain.value_objects.ids import FixtureId, LeagueId


def test_model_context_minimal_inputs() -> None:
    ctx = ModelContext(fixture_id=FixtureId(1), league_id=LeagueId(1), season=2024)
    assert not ctx.has_minimal_inputs_for(ModelName.POISSON)
    ctx_poisson = ctx.model_copy(update={"home_goal_rate": 1.0, "away_goal_rate": 1.0})
    assert ctx_poisson.has_minimal_inputs_for(ModelName.POISSON)
    ctx_elo = ctx.model_copy(update={"elo_home": 1.0, "elo_away": 1.0})
    assert ctx_elo.has_minimal_inputs_for(ModelName.ELO)
    ctx_lr = ctx.model_copy(update={"features": {"x": 0.5}})
    assert ctx_lr.has_minimal_inputs_for(ModelName.LOGISTIC_REGRESSION)
    assert ctx.has_minimal_inputs_for("something")


def test_strenum_backport(monkeypatch):
    module_name = "src.domain.interfaces.enums"
    mod = importlib.import_module(module_name)
    original_version = sys.version_info
    monkeypatch.setattr(sys, "version_info", (3, 10))
    mod = importlib.reload(mod)
    assert issubclass(mod.StrEnum, str)
    assert issubclass(mod.StrEnum, mod.Enum)
    monkeypatch.setattr(sys, "version_info", original_version)
    importlib.reload(mod)
