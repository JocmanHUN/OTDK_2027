from abc import ABC, abstractmethod
from typing import Optional

from src.domain.entities import Odds


class OddsRepo(ABC):
    @abstractmethod
    def create(self, entity: Odds) -> None:
        """
        Persist a new odds entry.

        Example:
            >>> repo.create(Odds(...))

        :param entity: Odds instance to persist.
        """

    @abstractmethod
    def get_by_id(self, odds_id: int) -> Optional[Odds]:
        """
        Fetch an odds record by its unique ID.

        Example:
            >>> repo.get_by_id(7)
            Odds(id=7, home=1.5, draw=3.8, away=5.0)

        :param odds_id: Unique identifier of the odds.
        :return: Odds object if found, otherwise None.
        """

    @abstractmethod
    def update(self, entity: Odds) -> None:
        """
        Update an existing odds record.

        Example:
            >>> repo.update(odds)

        :param entity: Odds instance with updated data.
        """

    @abstractmethod
    def delete(self, odds_id: int) -> None:
        """
        Delete an odds record by its ID.

        Example:
            >>> repo.delete(7)

        :param odds_id: Unique identifier of the odds to delete.
        """

    @abstractmethod
    def list_paginated(self, offset: int, limit: int) -> list[Odds]:
        """
        List odds in a paginated fashion.

        Example:
            >>> repo.list_paginated(0, 100)
            [Odds(...), Odds(...)]

        :param offset: Starting index.
        :param limit: Maximum number of odds records to return.
        :return: List of Odds entities.
        """
