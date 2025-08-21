from __future__ import annotations

from decimal import Decimal
from typing import ClassVar, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProbabilityTriplet(BaseModel):
    """Represents probabilities for home win, draw and away win."""

    home: float = Field(..., description="Probability of home win")
    draw: float = Field(..., description="Probability of draw")
    away: float = Field(..., description="Probability of away win")

    EPS: ClassVar[float] = 1e-9

    model_config = ConfigDict(frozen=True)

    @field_validator("home", "draw", "away")
    @classmethod
    def _prob_in_range(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Probability must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def _sum_to_one(self) -> "ProbabilityTriplet":
        total = self.home + self.draw + self.away
        if abs(total - 1) > self.EPS:
            raise ValueError("Probabilities must sum to 1 within tolerance")
        return self

    def normalized(self) -> "ProbabilityTriplet":
        """Return a new triplet scaled to sum to 1."""
        total = self.home + self.draw + self.away
        if total == 0:
            raise ValueError("Cannot normalize probabilities that sum to zero")
        if abs(total - 1) <= self.EPS:
            return self
        return ProbabilityTriplet(
            home=self.home / total, draw=self.draw / total, away=self.away / total
        )

    @classmethod
    def from_odds(
        cls, odds: Sequence[Decimal | float], remove_overround: bool = True
    ) -> "ProbabilityTriplet":
        if len(odds) != 3:
            raise ValueError("Exactly three odds are required")
        probs = [Decimal("1") / Decimal(str(o)) for o in odds]
        if remove_overround:
            total = sum(probs)
            probs = [p / total for p in probs]
        return cls(home=float(probs[0]), draw=float(probs[1]), away=float(probs[2]))

    def as_dict(self) -> dict[str, float]:
        return {"home": self.home, "draw": self.draw, "away": self.away}
