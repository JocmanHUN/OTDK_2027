from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Prediction:
    id: int | None
    match_id: int
    model_name: str
    prob_home: float
    prob_draw: float
    prob_away: float
    predicted_result: str
    is_correct: bool | None = None
    result_status: str = "PENDING"  # 'WIN', 'LOSE', or 'PENDING'


class PredictionsRepo(ABC):
    """Repository interface for predictions."""

    @abstractmethod
    def get_by_id(self, prediction_id: int) -> Optional[Prediction]:
        """Return a prediction by its identifier."""

    @abstractmethod
    def list_by_match(
        self, match_id: int, *, limit: int = 100, offset: int = 0
    ) -> list[Prediction]:
        """List predictions for a match with pagination."""

    @abstractmethod
    def insert(self, prediction: Prediction) -> int:
        """Persist a new prediction."""

    @abstractmethod
    def update(self, prediction: Prediction) -> None:
        """Update an existing prediction."""

    @abstractmethod
    def delete(self, prediction_id: int) -> None:
        """Remove a prediction by identifier."""

    @abstractmethod
    def mark_correct(self, prediction_id: int, is_correct: bool) -> None:
        """Mark the prediction as correct or not."""
