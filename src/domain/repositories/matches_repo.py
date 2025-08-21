from abc import ABC, abstractmethod
from typing import Optional

from src.domain.entities import Match


class MatchesRepo(ABC):
    @abstractmethod
    def create(self, entity: Match) -> None:
        """
        Persist a new match entity.

        Example:
            >>> repo.create(Match(...))

        :param entity: Match instance to persist.
        """

    @abstractmethod
    def get_by_id(self, match_id: int) -> Optional[Match]:
        """
        Fetch a match by its unique ID.

        Example:
            >>> repo.get_by_id(42)
            Match(id=42, home_name="Barcelona", away_name="Real Madrid")

        :param match_id: Unique identifier of the match.
        :return: Match object if found, otherwise None.
        """

    @abstractmethod
    def update(self, entity: Match) -> None:
        """
        Update an existing match.

        Example:
            >>> repo.update(match)

        :param entity: Match instance with updated data.
        """

    @abstractmethod
    def delete(self, match_id: int) -> None:
        """
        Delete a match by its ID.

        Example:
            >>> repo.delete(42)

        :param match_id: Unique identifier of the match to delete.
        """

    @abstractmethod
    def list_paginated(self, offset: int, limit: int) -> list[Match]:
        """
        List matches in a paginated fashion.

        Example:
            >>> repo.list_paginated(0, 50)
            [Match(...), Match(...)]

        :param offset: Starting index.
        :param limit: Maximum number of matches to return.
        :return: List of Match entities.
        """
