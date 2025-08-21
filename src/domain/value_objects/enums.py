from enum import Enum


class ModelName(str, Enum):
    POISSON = "poisson"
    MONTE_CARLO = "monte_carlo"
    ELO = "elo"
    LOGISTIC_REGRESSION = "logistic_regression"
    VETO = "veto"
    BALANCE = "balance"


class StrategyName(str, Enum):
    FLAT = "flat"
    MARTINGALE = "martingale"
    FIBONACCI = "fibonacci"
    VALUE = "value"
    KELLY = "kelly"


class Outcome(str, Enum):
    HOME = "HOME"
    DRAW = "DRAW"
    AWAY = "AWAY"
