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
    """Szimmetrikus Balance modell a megadott átlagoló képlettel.

    Paraméterek
    - `history`: provider with `get_recent_team_stats(...)` (tesztfüggő DI).
    - `last_n`: legutóbbi vizsgált meccsek száma (<=0 → üres történelem).
    - `decay_factor`: exponenciális súly 0 és 1 között (0 → 1).

    Kombináció
    - P(1) = 0.5 * (P_H(win) + (1 - P_A(win)))
    - P(2) = 0.5 * (P_A(win) + (1 - P_H(win)))
    - P(X) = 0.5 * (P_H(draw) + P_A(draw))
    A végeredményt 0-1 közé normalizáljuk (belsőleg, nem %-ban).
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

        p1_raw = 0.5 * (hW + (1.0 - aW))
        p2_raw = 0.5 * (aW + (1.0 - hW))
        px_raw = 0.5 * (hD + aD)

        # Normalize for numerical safety
        s = p1_raw + px_raw + p2_raw
        if s <= 0.0:
            probs = ProbabilityTriplet(home=1 / 3, draw=1 / 3, away=1 / 3)
        else:
            probs = ProbabilityTriplet(home=p1_raw / s, draw=px_raw / s, away=p2_raw / s)

        return Prediction(
            fixture_id=match.fixture_id,
            model=self.name,
            probs=probs,
            computed_at_utc=datetime.now(timezone.utc),
            version=self.version,
            status=PredictionStatus.OK,
        )
