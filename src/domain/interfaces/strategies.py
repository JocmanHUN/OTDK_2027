from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Protocol

from ..entities.odds import Odds
from ..entities.prediction import Prediction
from ..entities.strategy_result import StrategyResult
from ..value_objects.enums import Outcome, StrategyName
from ..value_objects.money import Money


class BaseStrategy(ABC):
    """Base class for betting strategies.

    >>> class Flat1Unit(BaseStrategy):
    ...     name = StrategyName.FLAT
    ...     version = "1.0"
    ...     def run(
    ...         self,
    ...         *,
    ...         bankroll: Money,
    ...         odds: Odds,
    ...         prediction: Prediction,
    ...         outcome: Outcome | None = None,
    ...         step_index: int = 0,
    ...     ) -> StrategyResult:
    ...         profit = Money(amount=0)
    ...         return StrategyResult(
    ...             fixture_id=odds.fixture_id,
    ...             strategy=self.name,
    ...             model=prediction.model,
    ...             outcome=Outcome.HOME,
    ...             stake=bankroll,
    ...             odds=odds.home,
    ...             result=None,
    ...             profit=profit,
    ...             bankroll_after=bankroll,
    ...             step_index=step_index,
    ...         )
    >>> isinstance(Flat1Unit().run, object)
    True
    """

    name: ClassVar[StrategyName]
    version: ClassVar[str]

    @abstractmethod
    def run(
        self,
        *,
        bankroll: Money,
        odds: Odds,
        prediction: Prediction,
        outcome: Outcome | None = None,
        step_index: int = 0,
    ) -> StrategyResult:
        """Execute one betting step and return the result.

        Negative profit is possible and indicates a loss.
        """


class StrategyRunner(Protocol):
    """Protocol for running a strategy on a sequence of events.

    >>> runner.simulate(strategy, seq, bankroll)  # doctest: +SKIP
    [StrategyResult(...)]
    """

    def simulate(
        self, strategy: BaseStrategy, sequence: list[tuple[Odds, Prediction]], bankroll: Money
    ) -> list[StrategyResult]: ...
