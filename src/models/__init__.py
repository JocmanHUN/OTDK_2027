from __future__ import annotations

from typing import List

from src.application.services.history_service import HistoryService
from src.domain.interfaces.modeling import BasePredictiveModel

from .balance import BalanceModel
from .balance_blend import BalanceBlendModel
from .balance_luck import BalanceLuckModel
from .balance_shift import BalanceShiftModel
from .elo import EloModel
from .logistic_regression import LogisticRegressionModel
from .monte_carlo import MonteCarloModel
from .poisson import PoissonModel
from .veto import VetoModel
from .veto_blend import VetoBlendModel
from .veto_luck import VetoLuckModel
from .veto_shift import VetoShiftModel


def default_models() -> List[BasePredictiveModel]:
    """Return the default set of predictive models (extend as new models are added)."""
    history = HistoryService()
    return [
        PoissonModel(),
        MonteCarloModel(),
        EloModel(),
        LogisticRegressionModel(),
        BalanceModel(history=history),
        BalanceBlendModel(history=history),
        BalanceLuckModel(history=history),
        BalanceShiftModel(history=history),
        VetoModel(history=history),
        VetoBlendModel(history=history),
        VetoLuckModel(history=history),
        VetoShiftModel(history=history),
    ]


def luck_variants(
    history: HistoryService | None = None,
) -> List[BasePredictiveModel]:
    """Convenience factory for low/medium/high luck variants without duplicating code."""
    history = history or HistoryService()
    presets = [
        ("low", 0.25, 0.9),
        ("medium", 0.5, 0.7),
        ("high", 0.75, 0.5),
    ]
    models: List[BasePredictiveModel] = []
    for tag, strength, threshold in presets:
        models.append(
            BalanceLuckModel(
                history=history,
                luck_strength=strength,
                luck_threshold=threshold,
                variant=tag,
            )
        )
        models.append(
            VetoLuckModel(
                history=history,
                luck_strength=strength,
                luck_threshold=threshold,
                variant=tag,
            )
        )
    return models


def blend_models(
    history: HistoryService | None = None, mix_weight: float = 0.5
) -> List[BasePredictiveModel]:
    """Return blend variants (shared alpha) for Balance/Veto."""
    history = history or HistoryService()
    return [
        BalanceBlendModel(history=history, mix_weight=mix_weight),
        VetoBlendModel(history=history, mix_weight=mix_weight),
    ]


def shift_models(
    history: HistoryService | None = None, margin_weight: float = 0.6, margin_clip: float = 2.0
) -> List[BasePredictiveModel]:
    """Return shift variants (shared shift params) for Balance/Veto."""
    history = history or HistoryService()
    return [
        BalanceShiftModel(history=history, margin_weight=margin_weight, margin_clip=margin_clip),
        VetoShiftModel(history=history, margin_weight=margin_weight, margin_clip=margin_clip),
    ]
