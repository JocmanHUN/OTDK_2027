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


def _form_distribution_with_luck(
    rows: list[dict],
    decay_factor: float,
    luck_strength: float,
    luck_threshold: float,
) -> tuple[float, float, float]:
    """Compute (win, draw, loss) distribution with xG luck adjustment."""
    if not rows:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    df = max(0.0, min(1.0, float(decay_factor)))
    if df == 0.0:
        df = 1.0

    ls = max(0.0, float(luck_strength))
    lt = max(1e-6, float(luck_threshold))

    w_win = w_draw = w_loss = 0.0
    w_sum = 0.0
    for idx, r in enumerate(rows):
        w = df**idx
        gf = int(r.get("goals_for") or 0)
        ga = int(r.get("goals_against") or 0)
        xgf = float(r.get("xg_for") or 0.0)
        xga = float(r.get("xg_against") or 0.0)

        luck_raw = (gf - ga) - (xgf - xga)
        luck_scaled = max(-1.0, min(1.0, luck_raw / lt))
        boost = max(0.05, 1.0 - ls * luck_scaled)

        if gf > ga:
            w_win += w * boost
        elif gf == ga:
            w_draw += w * boost
        else:
            w_loss += w * boost
        w_sum += w * boost

    if w_sum <= 0.0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    return (w_win / w_sum, w_draw / w_sum, w_loss / w_sum)


def _rows_have_xg(rows: list[dict]) -> bool:
    if not rows:
        return False
    for r in rows:
        if r.get("xg_for") is None or r.get("xg_against") is None:
            return False
    return True


@dataclass
class BalanceLuckModel(BasePredictiveModel):
    """Balance model with xG-based luck adjustment (downweights lucky, upweights unlucky)."""

    name: ClassVar[ModelName] = ModelName.BALANCE_LUCK
    version: ClassVar[str] = "2-luck"

    history: _HistoryProto | None = None
    last_n: int = 10
    decay_factor: float = 0.85
    luck_strength: float = 0.5
    luck_threshold: float = 0.7
    variant: str = "medium"

    def predict(self, match: Match, ctx: ModelContext) -> Prediction:
        if ctx.home_team_id is None or ctx.away_team_id is None:
            return Prediction(
                fixture_id=match.fixture_id,
                model=self.name,
                probs=None,
                computed_at_utc=datetime.now(timezone.utc),
                version=f"{self.version}-{self.variant}",
                status=PredictionStatus.SKIPPED,
                skip_reason="Missing team IDs in context",
            )

        if self.history is None:
            return Prediction(
                fixture_id=match.fixture_id,
                model=self.name,
                probs=None,
                computed_at_utc=datetime.now(timezone.utc),
                version=f"{self.version}-{self.variant}",
                status=PredictionStatus.SKIPPED,
                skip_reason="Missing history provider",
            )

        home_rows = self.history.get_recent_team_stats(
            int(ctx.home_team_id), int(ctx.league_id), int(ctx.season), self.last_n
        )
        away_rows = self.history.get_recent_team_stats(
            int(ctx.away_team_id), int(ctx.league_id), int(ctx.season), self.last_n
        )

        if not _rows_have_xg(home_rows) or not _rows_have_xg(away_rows):
            return Prediction(
                fixture_id=match.fixture_id,
                model=self.name,
                probs=None,
                computed_at_utc=datetime.now(timezone.utc),
                version=f"{self.version}-{self.variant}",
                status=PredictionStatus.SKIPPED,
                skip_reason="Missing xG data in history",
            )

        hW, hD, hL = _form_distribution_with_luck(
            home_rows, self.decay_factor, self.luck_strength, self.luck_threshold
        )
        aW, aD, aL = _form_distribution_with_luck(
            away_rows, self.decay_factor, self.luck_strength, self.luck_threshold
        )

        p1 = 0.5 * (hW + aL)
        px = 0.5 * (hD + aD)
        p2 = 0.5 * (aW + hL)

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
            version=f"{self.version}-{self.variant}",
            status=PredictionStatus.OK,
        )
