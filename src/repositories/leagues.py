from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class League:
    id: int | None
    name: str
    country: Optional[str] = None


class LeaguesRepo(ABC):
    """Repository interface for leagues."""

    @abstractmethod
    def get_by_id(self, league_id: int) -> Optional[League]:
        """Retrieve a league by identifier."""

    @abstractmethod
    def list_all(self, *, limit: int = 100, offset: int = 0) -> list[League]:
        """List all leagues with pagination."""

    @abstractmethod
    def insert(self, league: League) -> int:
        """Persist a new league."""

    @abstractmethod
    def update(self, league: League) -> None:
        """Update an existing league."""

    @abstractmethod
    def delete(self, league_id: int) -> None:
        """Remove a league by identifier."""
