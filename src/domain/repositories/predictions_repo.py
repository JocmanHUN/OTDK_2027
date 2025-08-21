from abc import ABC, abstractmethod
from typing import Optional

from src.domain.entities import Prediction


class PredictionsRepo(ABC):
    @abstractmethod
    def create(self, entity: Prediction) -> None:
        """
        Persist a new prediction.

        Example:
            >>> repo.create(Prediction(...))

        :param entity: Prediction instance to persist.
        """

    @abstractmethod
    def get_by_id(self, prediction_id: int) -> Optional[Prediction]:
        """
        Fetch a prediction by its unique ID.

        Example:
            >>> repo.get_by_id(3)
            Prediction(id=3, fixture_id=101, model="xgboost")

        :param prediction_id: Unique identifier of the prediction.
        :return: Prediction object if found, otherwise None.
        """

    @abstractmethod
    def update(self, entity: Prediction) -> None:
        """
        Update an existing prediction.

        Example:
            >>> repo.update(prediction)

        :param entity: Prediction instance with updated data.
        """

    @abstractmethod
    def delete(self, prediction_id: int) -> None:
        """
        Delete a prediction by its ID.

        Example:
            >>> repo.delete(3)

        :param prediction_id: Unique identifier of the prediction to delete.
        """

    @abstractmethod
    def list_paginated(self, offset: int, limit: int) -> list[Prediction]:
        """
        List predictions in a paginated fashion.

        Example:
            >>> repo.list_paginated(0, 20)
            [Prediction(...), Prediction(...)]

        :param offset: Starting index.
        :param limit: Maximum number of predictions to return.
        :return: List of Prediction entities.
        """
