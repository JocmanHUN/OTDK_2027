from __future__ import annotations

from typing import List

from src.domain.interfaces.modeling import BasePredictiveModel

from .elo import EloModel
from .logistic_regression import LogisticRegressionModel
from .monte_carlo import MonteCarloModel
from .poisson import PoissonModel


def default_models() -> List[BasePredictiveModel]:
    """Return the default set of predictive models.

    Extend this list as new models are implemented (total target: 6).
    """
    return [PoissonModel(), MonteCarloModel(), EloModel(), LogisticRegressionModel()]
