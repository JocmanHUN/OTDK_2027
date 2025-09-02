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
    """Compute weighted (win, draw, loss) distribution from recent matches.

    rows: list of dicts with keys 'goals_for' and 'goals_against'; most recent first.
    decay_factor: exponential decay for older matches in (0, 1].
    """
    if not rows:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    df = max(0.0, min(1.0, float(decay_factor)))
    if df == 0.0:
        df = 1.0  # treat as no decay to avoid zero weights

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
class BalanceModel(BasePredictiveModel):
    """Symmetric form-based model using exponential weighting over last N matches.

    Parameters
    - `history`: provider with `get_recent_team_stats(...)` (DI for testing).
    - `last_n`: number of recent matches to consider (<=0 yields uniform via empty history).
    - `decay_factor`: exponential decay per step back in time, clamped to [0,1]; 0 treated as 1.

    Combination
    - 1: mean(home.W, away.L)
    - X: mean(home.D, away.D)
    - 2: mean(away.W, home.L)
    - Normalized to sum to 1.
    """

    name: ClassVar[ModelName] = ModelName.BALANCE
    version: ClassVar[str] = "1"

    history: _HistoryProto | None = None
    last_n: int = 10
    decay_factor: float = 0.85

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

        # Fetch last N finished matches for both teams (league/season scoped)
        home_rows = self.history.get_recent_team_stats(
            int(ctx.home_team_id), int(ctx.league_id), int(ctx.season), self.last_n
        )
        away_rows = self.history.get_recent_team_stats(
            int(ctx.away_team_id), int(ctx.league_id), int(ctx.season), self.last_n
        )

        hW, hD, hL = _form_distribution(home_rows, self.decay_factor)
        aW, aD, aL = _form_distribution(away_rows, self.decay_factor)

        p1 = 0.5 * (hW + aL)
        px = 0.5 * (hD + aD)
        p2 = 0.5 * (aW + hL)

        # Normalize for numerical safety
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
