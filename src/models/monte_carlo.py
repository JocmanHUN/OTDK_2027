from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import ClassVar, Optional

import numpy as np

from src.domain.entities.match import Match
from src.domain.entities.prediction import Prediction
from src.domain.interfaces.context import ModelContext
from src.domain.interfaces.modeling import BasePredictiveModel
from src.domain.value_objects.enums import ModelName, PredictionStatus
from src.domain.value_objects.probability_triplet import ProbabilityTriplet


@dataclass
class MonteCarloModel(BasePredictiveModel):
    """Monte Carlo model simulating scorelines from Poisson goal rates.

    Uses `ctx.home_goal_rate` and `ctx.away_goal_rate` as Poisson lambdas.
    """

    version: ClassVar[str] = "1"
    n_sims: int = 20000
    random_seed: Optional[int] = None

    # Name is a ClassVar on the base, but keeping attribute for clarity in repr
    name: ClassVar[ModelName] = ModelName.MONTE_CARLO

    def predict(self, match: Match, ctx: ModelContext) -> Prediction:
        if ctx.home_goal_rate is None or ctx.away_goal_rate is None:
            return Prediction(
                fixture_id=match.fixture_id,
                model=self.name,
                probs=None,
                computed_at_utc=datetime.now(timezone.utc),
                version=self.version,
                status=PredictionStatus.SKIPPED,
                skip_reason="Missing home/away goal rates",
            )

        lam_h = float(ctx.home_goal_rate)
        lam_a = float(ctx.away_goal_rate)

        rng = np.random.default_rng(self.random_seed)
        h = rng.poisson(lam=lam_h, size=int(self.n_sims))
        a = rng.poisson(lam=lam_a, size=int(self.n_sims))

        home_wins = int(np.count_nonzero(h > a))
        away_wins = int(np.count_nonzero(h < a))
        # draws is implied as total - home - away
        total = float(self.n_sims)

        # Normalize defensively (avoid strict validator issues)
        p_home = home_wins / total
        p_away = away_wins / total
        p_draw = max(0.0, 1.0 - p_home - p_away)

        probs = ProbabilityTriplet(home=p_home, draw=p_draw, away=p_away).normalized()

        return Prediction(
            fixture_id=match.fixture_id,
            model=self.name,
            probs=probs,
            computed_at_utc=datetime.now(timezone.utc),
            version=self.version,
            status=PredictionStatus.OK,
        )
