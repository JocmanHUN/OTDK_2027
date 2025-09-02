from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp
from typing import ClassVar, Mapping

from src.domain.entities.match import Match
from src.domain.entities.prediction import Prediction
from src.domain.interfaces.context import ModelContext
from src.domain.interfaces.modeling import BasePredictiveModel
from src.domain.value_objects.enums import ModelName, PredictionStatus
from src.domain.value_objects.probability_triplet import ProbabilityTriplet


def _sigmoid(x: float) -> float:
    try:
        if x >= 0:
            z = exp(-x)
            return 1.0 / (1.0 + z)
        z = exp(x)
        return z / (1.0 + z)
    except OverflowError:
        return 0.0 if x < 0 else 1.0


@dataclass
class LogisticRegressionModel(BasePredictiveModel):
    """Stats-driven logistic model using recent feature differences.

    Parameters
    - `base_draw`: baseline draw probability at neutral score (clamped within [0, 0.9] at use time).
    - `draw_sensitivity`: how strongly to increase draw when the matchup is close.

    Inputs
    - `ctx.features`: mapping with `diff_*` feature names (from `FeaturesService`). Missing -> SKIPPED.
    """

    version: ClassVar[str] = "1"
    base_draw: float = 0.28
    draw_sensitivity: float = 0.35  # how much to boost draw when close

    name: ClassVar[ModelName] = ModelName.LOGISTIC_REGRESSION

    def _weights(self) -> Mapping[str, float]:
        # Heuristic weights; can be learned later.
        return {
            "diff_goals_for_avg": 0.60,
            "diff_goals_against_avg": -0.55,
            "diff_points_per_game": 0.70,
            "diff_shots on target": 0.08,
            "diff_shots": 0.04,
            "diff_ball possession": 0.03,
            "diff_corners": 0.02,
            "diff_big chances": 0.06,
            "diff_fouls": -0.02,
            "diff_yellow cards": -0.05,
            "diff_red cards": -0.30,
            # Add more as needed; keys must match API stat names normalized in HistoryService
        }

    def predict(self, match: Match, ctx: ModelContext) -> Prediction:
        feats = ctx.features
        if not isinstance(feats, Mapping) or not feats:
            return Prediction(
                fixture_id=match.fixture_id,
                model=self.name,
                probs=None,
                computed_at_utc=datetime.now(timezone.utc),
                version=self.version,
                status=PredictionStatus.SKIPPED,
                skip_reason="Missing features",
            )

        w = self._weights()
        score = 0.0
        used = set()
        for k, weight in w.items():
            v = float(feats.get(k, 0.0))
            score += weight * v
            used.add(k)

        # Light-touch inclusion of remaining stats with tiny heuristic weights
        for k, v_raw in feats.items():
            if k in used or not k.startswith("diff_"):
                continue
            try:
                v = float(v_raw)
            except (TypeError, ValueError):
                continue
            kl = k.lower()
            if any(
                s in kl for s in ["foul", "red card", "yellow card", "offside", "error", "own goal"]
            ):
                weight = -0.01
            else:
                weight = 0.01
            score += weight * v

        p_home_raw = _sigmoid(score)

        # Draw probability higher for close games around 0.5
        closeness = 1.0 - abs(p_home_raw - 0.5) * 2.0  # in [0,1]
        base_draw = min(0.9, max(0.0, float(self.base_draw)))
        p_draw = max(0.0, min(0.6, base_draw + float(self.draw_sensitivity) * closeness))

        p_no_draw = max(1e-9, 1.0 - p_draw)
        p_home = p_no_draw * p_home_raw
        p_away = p_no_draw * (1.0 - p_home_raw)

        probs = ProbabilityTriplet(home=p_home, draw=p_draw, away=p_away).normalized()
        return Prediction(
            fixture_id=match.fixture_id,
            model=self.name,
            probs=probs,
            computed_at_utc=datetime.now(timezone.utc),
            version=self.version,
            status=PredictionStatus.OK,
        )
