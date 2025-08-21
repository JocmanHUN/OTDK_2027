from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Bookmaker:
    id: int | None
    name: str


class BookmakersRepo(ABC):
    """Repository interface for bookmakers."""

    @abstractmethod
    def get_by_id(self, bookmaker_id: int) -> Optional[Bookmaker]:
        """Retrieve a bookmaker by identifier."""

    @abstractmethod
    def list_all(self, *, limit: int = 100, offset: int = 0) -> list[Bookmaker]:
        """List all bookmakers with pagination."""

    @abstractmethod
    def insert(self, bookmaker: Bookmaker) -> int:
        """Persist a new bookmaker."""

    @abstractmethod
    def update(self, bookmaker: Bookmaker) -> None:
        """Update an existing bookmaker."""

    @abstractmethod
    def delete(self, bookmaker_id: int) -> None:
        """Remove a bookmaker by identifier."""
