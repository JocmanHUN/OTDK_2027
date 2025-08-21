from abc import ABC, abstractmethod
from typing import Optional

from src.domain.entities import StrategyResult


class SimulationsRepo(ABC):
    @abstractmethod
    def create(self, entity: StrategyResult) -> None:
        """
        Persist a new strategy result.

        Example:
            >>> repo.create(StrategyResult(...))

        :param entity: StrategyResult instance to persist.
        """

    @abstractmethod
    def get_by_id(self, result_id: int) -> Optional[StrategyResult]:
        """
        Fetch a simulation result by its unique ID.

        Example:
            >>> repo.get_by_id(5)
            StrategyResult(id=5, outcome="WIN")

        :param result_id: Unique identifier of the simulation result.
        :return: StrategyResult object if found, otherwise None.
        """

    @abstractmethod
    def update(self, entity: StrategyResult) -> None:
        """
        Update an existing simulation result.

        Example:
            >>> repo.update(result)

        :param entity: StrategyResult instance with updated data.
        """

    @abstractmethod
    def delete(self, result_id: int) -> None:
        """
        Delete a simulation result by its ID.

        Example:
            >>> repo.delete(5)

        :param result_id: Unique identifier of the simulation result to delete.
        """

    @abstractmethod
    def list_paginated(self, offset: int, limit: int) -> list[StrategyResult]:
        """
        List simulation results in a paginated fashion.

        Example:
            >>> repo.list_paginated(0, 10)
            [StrategyResult(...), StrategyResult(...)]

        :param offset: Starting index.
        :param limit: Maximum number of simulation results to return.
        :return: List of StrategyResult entities.
        """
