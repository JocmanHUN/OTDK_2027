from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..value_objects.enums import ModelName, Outcome, StrategyName
from ..value_objects.ids import FixtureId
from ..value_objects.money import Money


class StrategyResult(BaseModel):
    fixture_id: FixtureId
    strategy: StrategyName
    model: ModelName
    outcome: Outcome
    stake: Money
    odds: Decimal
    result: Literal["WIN", "LOSE", "VOID"] | None = None
    profit: Money
    bankroll_after: Money
    step_index: int = Field(..., ge=0)
    notes: str | None = None

    model_config = ConfigDict(frozen=True)
