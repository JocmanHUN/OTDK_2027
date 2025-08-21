from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Sequence, Tuple

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ..value_objects.enums import ModelName, Outcome, PredictionStatus
from ..value_objects.ids import BookmakerId, FixtureId
from ..value_objects.probability_triplet import ProbabilityTriplet
from .odds import Odds


class Prediction(BaseModel):
    fixture_id: FixtureId
    model: ModelName
    probs: ProbabilityTriplet | None = None
    computed_at_utc: datetime
    version: str
    status: PredictionStatus
    skip_reason: str | None = None

    model_config = ConfigDict(frozen=True)

    @field_validator("computed_at_utc", mode="before")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("computed_at_utc must be timezone-aware")
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _check_status_requirements(self) -> "Prediction":
        if self.status == PredictionStatus.OK and self.probs is None:
            raise ValueError("probs must be provided when status is OK")
        if self.status == PredictionStatus.SKIPPED and not self.skip_reason:
            raise ValueError("skip_reason must be provided when status is SKIPPED")
        return self

    def ev(self, odds: Odds, outcome: Outcome) -> Decimal:
        if self.probs is None:
            raise ValueError("Cannot compute EV without probabilities")
        prob_map = {
            Outcome.HOME: self.probs.home,
            Outcome.DRAW: self.probs.draw,
            Outcome.AWAY: self.probs.away,
        }
        odds_map = {Outcome.HOME: odds.home, Outcome.DRAW: odds.draw, Outcome.AWAY: odds.away}
        probability = Decimal(str(prob_map[outcome]))
        return probability * odds_map[outcome] - Decimal("1")

    def best_ev(self, odds_candidates: Sequence[Odds]) -> Tuple[Outcome, BookmakerId, Decimal]:
        best_outcome: Outcome | None = None
        best_bookmaker: BookmakerId | None = None
        best_value = Decimal("-Infinity")
        for o in odds_candidates:
            for outcome in Outcome:
                value = self.ev(o, outcome)
                if value > best_value:
                    best_value = value
                    best_outcome = outcome
                    best_bookmaker = o.bookmaker_id
        assert best_outcome is not None and best_bookmaker is not None
        return best_outcome, best_bookmaker, best_value
