from __future__ import annotations

from typing import List

from src.application.services.history_service import HistoryService
from src.domain.interfaces.modeling import BasePredictiveModel

from .balance import BalanceModel
from .elo import EloModel
from .logistic_regression import LogisticRegressionModel
from .monte_carlo import MonteCarloModel
from .poisson import PoissonModel
from .veto import VetoModel


def default_models() -> List[BasePredictiveModel]:
    """Return the default set of predictive models.

    Extend this list as new models are implemented (total target: 6).
    """
    history = HistoryService()
    return [
        PoissonModel(),
        MonteCarloModel(),
        EloModel(),
        LogisticRegressionModel(),
        BalanceModel(history=history),
        VetoModel(history=history),
    ]
