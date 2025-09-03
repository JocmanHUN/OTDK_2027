from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Match:
    """Lightweight representation of a football match."""

    id: int | None
    league_id: int
    season: int
    date: datetime
    home_team: str
    away_team: str
    real_result: str | None = None  # '1', 'X' or '2'


class MatchesRepo(ABC):
    """Abstract repository interface for :class:`Match` entities."""

    @abstractmethod
    def get_by_id(self, match_id: int) -> Optional[Match]:
        """Return a match by its identifier if present."""

    @abstractmethod
    def list_by_league(
        self, league_id: int, season: int, *, limit: int = 100, offset: int = 0
    ) -> list[Match]:
        """List matches for a league and season with pagination."""

    @abstractmethod
    def insert(self, match: Match) -> int:
        """Persist a new match and return the assigned identifier."""

    @abstractmethod
    def update_result(self, match_id: int, real_result: str) -> None:
        """Update the real result of a match."""

    @abstractmethod
    def delete(self, match_id: int) -> None:
        """Remove a match by its identifier."""
