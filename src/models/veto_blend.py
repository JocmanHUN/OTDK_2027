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


def _form_xg_distribution(
    rows: list[dict], decay_factor: float, xg_threshold: float
) -> tuple[float, float, float]:
    if not rows:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    df = max(0.0, min(1.0, float(decay_factor)))
    if df == 0.0:
        df = 1.0

    th = max(1e-6, float(xg_threshold))

    w_win = w_draw = w_loss = 0.0
    w_sum = 0.0
    for idx, r in enumerate(rows):
        w = df**idx
        diff = float(r.get("xg_for") or 0.0) - float(r.get("xg_against") or 0.0)
        if diff > th:
            w_win += w
        elif diff < -th:
            w_loss += w
        else:
            w_draw += w
        w_sum += w

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
class VetoBlendModel(BasePredictiveModel):
    """Veto model that blends result-form with xG-form via alpha mixing (default/low draw sensitivity)."""

    name: ClassVar[ModelName] = ModelName.VETO_BLEND
    version: ClassVar[str] = "2-blend"

    history: _HistoryProto | None = None
    last_n: int = 10
    decay_factor: float = 0.85
    mul_weight: float = 0.6
    mix_weight: float = 0.3  # alpha between xG-form and result-form (keeps result-form dominant)
    xg_threshold: float = 0.1  # tighter margin to avoid draw-heavy bias

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

        if not _rows_have_xg(home_rows) or not _rows_have_xg(away_rows):
            return Prediction(
                fixture_id=match.fixture_id,
                model=self.name,
                probs=None,
                computed_at_utc=datetime.now(timezone.utc),
                version=self.version,
                status=PredictionStatus.SKIPPED,
                skip_reason="Missing xG data in history",
            )

        hW_r, hD_r, hL_r = _form_distribution(home_rows, self.decay_factor)
        aW_r, aD_r, aL_r = _form_distribution(away_rows, self.decay_factor)

        hW_x, hD_x, hL_x = _form_xg_distribution(home_rows, self.decay_factor, self.xg_threshold)
        aW_x, aD_x, aL_x = _form_xg_distribution(away_rows, self.decay_factor, self.xg_threshold)

        alpha = max(0.0, min(1.0, float(self.mix_weight)))

        hW = alpha * hW_x + (1 - alpha) * hW_r
        hD = alpha * hD_x + (1 - alpha) * hD_r
        hL = alpha * hL_x + (1 - alpha) * hL_r

        aW = alpha * aW_x + (1 - alpha) * aW_r
        aD = alpha * aD_x + (1 - alpha) * aD_r
        aL = alpha * aL_x + (1 - alpha) * aL_r

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


@dataclass
class VetoBlendMediumModel(VetoBlendModel):
    """Veto blend variant with a wider draw band and softer product weight."""

    name: ClassVar[ModelName] = ModelName.VETO_BLEND_MEDIUM
    version: ClassVar[str] = "2-blend-med"

    xg_threshold: float = 0.2
    mix_weight: float = 0.25
    mul_weight: float = 0.5


@dataclass
class VetoBlendHighModel(VetoBlendModel):
    """Veto blend variant most inclined to keep draws alive."""

    name: ClassVar[ModelName] = ModelName.VETO_BLEND_HIGH
    version: ClassVar[str] = "2-blend-high"

    xg_threshold: float = 0.25
    mix_weight: float = 0.35
    mul_weight: float = 0.4
