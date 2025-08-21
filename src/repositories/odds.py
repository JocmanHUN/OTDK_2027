from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Odds:
    id: int | None
    match_id: int
    bookmaker_id: int
    home: float
    draw: float
    away: float


class OddsRepo(ABC):
    """Repository interface for odds records."""

    @abstractmethod
    def get_by_id(self, odds_id: int) -> Optional[Odds]:
        """Return odds by identifier if present."""

    @abstractmethod
    def list_by_match(self, match_id: int, *, limit: int = 100, offset: int = 0) -> list[Odds]:
        """List odds entries for a given match."""

    @abstractmethod
    def insert(self, odds: Odds) -> int:
        """Persist new odds and return the identifier."""

    @abstractmethod
    def update(self, odds: Odds) -> None:
        """Update an existing odds record."""

    @abstractmethod
    def delete(self, odds_id: int) -> None:
        """Remove odds by identifier."""

    @abstractmethod
    def best_odds(self, match_id: int) -> dict[str, tuple[int, float]]:
        """Return the best odds per outcome for a match."""
