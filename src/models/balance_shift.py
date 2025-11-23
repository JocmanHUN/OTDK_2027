from __future__ import annotations

import math
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


def _xg_margin(rows: list[dict], decay_factor: float) -> float:
    """Return weighted mean xG margin; requires xg_for/xg_against."""
    if not rows:
        return 0.0

    df = max(0.0, min(1.0, float(decay_factor)))
    if df == 0.0:
        df = 1.0

    w_sum = 0.0
    margin_sum = 0.0
    for idx, r in enumerate(rows):
        w = df**idx
        xgf = float(r.get("xg_for") or 0.0)
        xga = float(r.get("xg_against") or 0.0)
        margin = xgf - xga
        margin_sum += w * margin
        w_sum += w

    if w_sum <= 0.0:
        return 0.0
    return margin_sum / w_sum


def _softmax_shift(p1: float, px: float, p2: float, shift: float) -> tuple[float, float, float]:
    eps = 1e-12
    log1 = math.log(max(eps, p1)) + shift
    logx = math.log(max(eps, px))
    log2 = math.log(max(eps, p2)) - shift
    m = max(log1, logx, log2)
    e1 = math.exp(log1 - m)
    ex = math.exp(logx - m)
    e2 = math.exp(log2 - m)
    s = e1 + ex + e2
    if s <= 0.0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (e1 / s, ex / s, e2 / s)


@dataclass
class BalanceShiftModel(BasePredictiveModel):
    """Balance model with xG margin shift applied to final probabilities."""

    name: ClassVar[ModelName] = ModelName.BALANCE_SHIFT
    version: ClassVar[str] = "2-shift"

    history: _HistoryProto | None = None
    last_n: int = 10
    decay_factor: float = 0.85
    margin_weight: float = 0.6
    margin_clip: float = 2.0

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

        if not home_rows or not away_rows:
            return Prediction(
                fixture_id=match.fixture_id,
                model=self.name,
                probs=None,
                computed_at_utc=datetime.now(timezone.utc),
                version=self.version,
                status=PredictionStatus.SKIPPED,
                skip_reason="Missing xG data in history",
            )
        if any(
            r.get("xg_for") is None or r.get("xg_against") is None for r in home_rows + away_rows
        ):
            return Prediction(
                fixture_id=match.fixture_id,
                model=self.name,
                probs=None,
                computed_at_utc=datetime.now(timezone.utc),
                version=self.version,
                status=PredictionStatus.SKIPPED,
                skip_reason="Missing xG data in history",
            )

        hW, hD, hL = _form_distribution(home_rows, self.decay_factor)
        aW, aD, aL = _form_distribution(away_rows, self.decay_factor)

        p1 = 0.5 * (hW + aL)
        px = 0.5 * (hD + aD)
        p2 = 0.5 * (aW + hL)

        home_margin = _xg_margin(home_rows, self.decay_factor)
        away_margin = _xg_margin(away_rows, self.decay_factor)
        shift = self.margin_weight * (home_margin - away_margin)
        shift = max(-float(self.margin_clip), min(float(self.margin_clip), shift))

        ph, pd, pa = _softmax_shift(p1, px, p2, shift)

        probs = ProbabilityTriplet(home=ph, draw=pd, away=pa)
        return Prediction(
            fixture_id=match.fixture_id,
            model=self.name,
            probs=probs,
            computed_at_utc=datetime.now(timezone.utc),
            version=self.version,
            status=PredictionStatus.OK,
        )
