from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import ClassVar, Protocol

from src.domain.entities.match import Match
from src.domain.entities.prediction import Prediction
from src.domain.interfaces.context import ModelContext
from src.domain.interfaces.modeling import BasePredictiveModel
from src.domain.value_objects.enums import ModelName, PredictionStatus
from src.domain.value_objects.probability_triplet import ProbabilityTriplet


class _HistoryProto(Protocol):
    def get_recent_team_stats(
        self,
        team_id: int,
        league_id: int,
        season: int,
        last: int,
        *,
        only_finished: bool = True,
    ) -> list[dict]: ...  # pragma: no cover


def _form_distribution(rows: list[dict], decay_factor: float) -> tuple[float, float, float]:
    if not rows:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    df = max(0.0, min(1.0, float(decay_factor)))
    if df == 0.0:
        df = 1.0

    w_win = w_draw = w_loss = 0.0
    w_sum = 0.0
    for idx, r in enumerate(rows):
        w = df**idx
        gf = int(r.get("goals_for") or 0)
        ga = int(r.get("goals_against") or 0)
        if gf > ga:
            w_win += w
        elif gf == ga:
            w_draw += w
        else:
            w_loss += w
        w_sum += w

    if w_sum <= 0.0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    return (w_win / w_sum, w_draw / w_sum, w_loss / w_sum)


@dataclass
class VetoModel(BasePredictiveModel):
    """Asymmetric form combiner with product and weighted-average components.

    Raw components:
      1: hW * aL  and  avg(hW, aL)
      X: hD * aD  and  avg(hD, aD)
      2: aW * hL  and  avg(aW, hL)
    Final raw score = mul_weight * product + (1 - mul_weight) * average.
    """

    name: ClassVar[ModelName] = ModelName.VETO
    version: ClassVar[str] = "1"

    history: _HistoryProto | None = None
    last_n: int = 10
    decay_factor: float = 0.85
    mul_weight: float = 0.6  # emphasis on the veto-like multiplicative agreement

    def predict(self, match: Match, ctx: ModelContext) -> Prediction:
        if ctx.home_team_id is None or ctx.away_team_id is None:
            return Prediction(
                fixture_id=match.fixture_id,
                model=self.name,
                probs=None,
                computed_at_utc=datetime.now(timezone.utc),
                version=self.version,
                status=PredictionStatus.SKIPPED,
                skip_reason="Missing team IDs in context",
            )

        if self.history is None:
            return Prediction(
                fixture_id=match.fixture_id,
                model=self.name,
                probs=None,
                computed_at_utc=datetime.now(timezone.utc),
                version=self.version,
                status=PredictionStatus.SKIPPED,
                skip_reason="Missing history provider",
            )

        home_rows = self.history.get_recent_team_stats(
            int(ctx.home_team_id), int(ctx.league_id), int(ctx.season), self.last_n
        )
        away_rows = self.history.get_recent_team_stats(
            int(ctx.away_team_id), int(ctx.league_id), int(ctx.season), self.last_n
        )

        hW, hD, hL = _form_distribution(home_rows, self.decay_factor)
        aW, aD, aL = _form_distribution(away_rows, self.decay_factor)

        w = max(0.0, min(1.0, float(self.mul_weight)))

        p1_prod = hW * aL
        p1_avg = 0.5 * (hW + aL)
        p1 = w * p1_prod + (1.0 - w) * p1_avg

        px_prod = hD * aD
        px_avg = 0.5 * (hD + aD)
        px = w * px_prod + (1.0 - w) * px_avg

        p2_prod = aW * hL
        p2_avg = 0.5 * (aW + hL)
        p2 = w * p2_prod + (1.0 - w) * p2_avg

        s = p1 + px + p2
        if s <= 0.0:
            probs = ProbabilityTriplet(home=1 / 3, draw=1 / 3, away=1 / 3)
        else:
            probs = ProbabilityTriplet(home=p1 / s, draw=px / s, away=p2 / s)

        return Prediction(
            fixture_id=match.fixture_id,
            model=self.name,
            probs=probs,
            computed_at_utc=datetime.now(timezone.utc),
            version=self.version,
            status=PredictionStatus.OK,
        )
