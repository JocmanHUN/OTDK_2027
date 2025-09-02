from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import ClassVar, Iterable

from src.domain.entities.match import Match
from src.domain.entities.prediction import Prediction
from src.domain.interfaces.context import ModelContext
from src.domain.interfaces.modeling import BasePredictiveModel
from src.domain.value_objects.enums import ModelName, PredictionStatus
from src.domain.value_objects.probability_triplet import ProbabilityTriplet


def _poisson_pmf_vector(mu: float, *, tol: float = 1e-8, max_k: int = 15) -> list[float]:
    """Return Poisson PMF values for k = 0..N where tail mass < tol or N == max_k.

    Uses the recurrence p[k+1] = p[k] * mu / (k+1) for numerical stability.
    """
    mu = float(max(mu, 0.0))
    # Handle degenerate case
    if mu == 0.0:
        return [1.0]

    # Start with k=0 term: e^{-mu}
    p0 = math.exp(-mu)
    probs = [p0]
    cumulative = p0
    k = 0
    while cumulative < (1.0 - tol) and k < max_k:
        pk = probs[-1] * mu / float(k + 1)
        probs.append(pk)
        cumulative += pk
        k += 1

    return probs


def _outer_sum(p_home: Iterable[float], p_away: Iterable[float]) -> tuple[float, float, float]:
    """Compute (P(home win), P(draw), P(away win)) from independent goal PMFs."""
    ph = list(p_home)
    pa = list(p_away)
    n_h = len(ph)
    n_a = len(pa)
    p_home_win = 0.0
    p_draw = 0.0
    p_away_win = 0.0

    for i in range(n_h):
        for j in range(n_a):
            pij = ph[i] * pa[j]
            if i > j:
                p_home_win += pij
            elif i == j:
                p_draw += pij
            else:
                p_away_win += pij
    return p_home_win, p_draw, p_away_win


@dataclass
class PoissonModel(BasePredictiveModel):
    """Independent-goals Poisson model.

    Parameters
    - `tol`: tail mass tolerance when truncating the Poisson distribution (min 1e-12).
    - `max_goals`: maximum goals considered per team (min 0).

    Inputs
    - `ctx.home_goal_rate`, `ctx.away_goal_rate`: non-negative Poisson lambdas; negatives are treated as 0.
    """

    name: ClassVar[ModelName] = ModelName.POISSON
    version: ClassVar[str] = "1"
    tol: float = 1e-6
    max_goals: int = 15

    def predict(self, match: Match, ctx: ModelContext) -> Prediction:
        if not ctx.has_minimal_inputs_for(ModelName.POISSON):
            return Prediction(
                fixture_id=match.fixture_id,
                model=ModelName.POISSON,
                probs=None,
                computed_at_utc=datetime.now(timezone.utc),
                version=self.version,
                status=PredictionStatus.SKIPPED,
                skip_reason="Missing home/away goal rates",
            )

        assert ctx.home_goal_rate is not None and ctx.away_goal_rate is not None
        # Build goal PMFs
        tol = max(1e-12, float(self.tol))
        max_k = max(0, int(self.max_goals))
        p_h = _poisson_pmf_vector(float(ctx.home_goal_rate), tol=tol, max_k=max_k)
        p_a = _poisson_pmf_vector(float(ctx.away_goal_rate), tol=tol, max_k=max_k)

        p_home, p_draw, p_away = _outer_sum(p_h, p_a)
        total = p_home + p_draw + p_away
        if total <= 0:
            p_home = p_draw = p_away = 1.0 / 3.0
        else:
            p_home /= total
            p_draw /= total
            p_away /= total
        probs = ProbabilityTriplet(home=p_home, draw=p_draw, away=p_away)

        return Prediction(
            fixture_id=match.fixture_id,
            model=ModelName.POISSON,
            probs=probs,
            computed_at_utc=datetime.now(timezone.utc),
            version=self.version,
            status=PredictionStatus.OK,
        )
