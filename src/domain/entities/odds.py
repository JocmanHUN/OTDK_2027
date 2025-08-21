from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from ..value_objects.enums import Outcome
from ..value_objects.ids import BookmakerId, FixtureId
from ..value_objects.probability_triplet import ProbabilityTriplet


class Odds(BaseModel):
    fixture_id: FixtureId
    bookmaker_id: BookmakerId
    collected_at_utc: datetime
    home: Decimal
    draw: Decimal
    away: Decimal
    market: Literal["1x2"] = "1x2"

    model_config = ConfigDict(frozen=True)

    @field_validator("collected_at_utc", mode="before")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("collected_at_utc must be timezone-aware")
        return v.astimezone(timezone.utc)

    @field_validator("home", "draw", "away")
    @classmethod
    def _check_minimum(cls, v: Decimal) -> Decimal:
        if v < Decimal("1.01"):
            raise ValueError("Odds must be at least 1.01")
        return v

    def implied_probabilities(self) -> ProbabilityTriplet:
        odds = [self.home, self.draw, self.away]
        return ProbabilityTriplet.from_odds(odds)


def best_of(odds_list: Sequence[Odds]) -> dict[Outcome, tuple[BookmakerId, Decimal]]:
    best: dict[Outcome, tuple[BookmakerId, Decimal]] = {}
    for o in odds_list:
        if (Outcome.HOME not in best) or (o.home > best[Outcome.HOME][1]):
            best[Outcome.HOME] = (o.bookmaker_id, o.home)
        if (Outcome.DRAW not in best) or (o.draw > best[Outcome.DRAW][1]):
            best[Outcome.DRAW] = (o.bookmaker_id, o.draw)
        if (Outcome.AWAY not in best) or (o.away > best[Outcome.AWAY][1]):
            best[Outcome.AWAY] = (o.bookmaker_id, o.away)
    return best
