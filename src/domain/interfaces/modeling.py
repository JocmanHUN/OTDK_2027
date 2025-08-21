from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Protocol

from ..entities.match import Match
from ..entities.prediction import Prediction
from ..value_objects.enums import ModelName
from .context import ModelContext


class BasePredictiveModel(ABC):
    """Base class for predictive models.

    >>> from datetime import datetime, timezone
    >>> from src.domain.interfaces.enums import PredictionStatus
    >>> from src.domain.value_objects.probability_triplet import ProbabilityTriplet
    >>> class AlwaysHome(BasePredictiveModel):
    ...     name = ModelName.POISSON
    ...     version = "0"
    ...     def predict(self, match: Match, ctx: ModelContext) -> Prediction:
    ...         probs = ProbabilityTriplet(home=1.0, draw=0.0, away=0.0)
    ...         return Prediction(
    ...             fixture_id=match.fixture_id,
    ...             model=self.name,
    ...             probs=probs,
    ...             computed_at_utc=datetime.now(timezone.utc),
    ...             version=self.version,
    ...             status=PredictionStatus.OK,
    ...         )
    >>> isinstance(AlwaysHome().predict, object)
    True
    """

    name: ClassVar[ModelName]
    version: ClassVar[str]

    @abstractmethod
    def predict(self, match: Match, ctx: ModelContext) -> Prediction:
        """Return a prediction for ``match`` using ``ctx``.

        When required inputs are missing, implementations should return a
        ``Prediction`` with ``status=PredictionStatus.SKIPPED`` and provide a
        ``skip_reason`` instead of raising exceptions.
        """


class PredictionAggregator(Protocol):
    """Protocol for running multiple predictive models.

    >>> agg.run_all([model], match, ctx)  # doctest: +SKIP
    [Prediction(...)]
    """

    def run_all(
        self, models: list[BasePredictiveModel], match: Match, ctx: ModelContext
    ) -> list[Prediction]: ...
