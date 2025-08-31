from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from typing import ClassVar, Optional

from src.application.services.elo_service import EloParams, EloService
from src.config.league_tiers import get_tier_config
from src.domain.entities.match import Match
from src.domain.entities.prediction import Prediction
from src.domain.interfaces.context import ModelContext
from src.domain.interfaces.modeling import BasePredictiveModel
from src.domain.value_objects.enums import ModelName, PredictionStatus
from src.domain.value_objects.probability_triplet import ProbabilityTriplet


def _davidson_probs(delta_with_home: float, nu: float) -> tuple[float, float, float]:
    # gamma = 10^(Δ/400)
    gamma = 10.0 ** (delta_with_home / 400.0)
    root = sqrt(gamma)
    denom = gamma + 1.0 + 2.0 * nu * root
    if denom <= 0.0:
        # Fallback safe uniform
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    p_home = gamma / denom
    p_draw = (2.0 * nu * root) / denom
    p_away = 1.0 / denom
    return p_home, p_draw, p_away


@dataclass
class EloModel(BasePredictiveModel):
    version: ClassVar[str] = "1"
    draw_param: Optional[float] = None  # if None, use EloParams default
    elo_service: Optional[EloService] = None

    # Base name
    name: ClassVar[ModelName] = ModelName.ELO

    def predict(self, match: Match, ctx: ModelContext) -> Prediction:
        league_id = int(ctx.league_id)
        season = int(ctx.season)
        cfg = get_tier_config(league_id)

        svc = self.elo_service or EloService()

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

        # Use context ELOs if present; otherwise compute via service
        if ctx.elo_home is not None and ctx.elo_away is not None:
            r_home, r_away = float(ctx.elo_home), float(ctx.elo_away)
        else:
            r_home = svc.get_team_rating(league_id, season, int(ctx.home_team_id))
            r_away = svc.get_team_rating(league_id, season, int(ctx.away_team_id))

        delta = r_home - r_away + float(cfg.home_adv)
        nu = float(self.draw_param if self.draw_param is not None else EloParams().draw_param)
        p_home, p_draw, p_away = _davidson_probs(delta, nu)
        probs = ProbabilityTriplet(home=p_home, draw=p_draw, away=p_away).normalized()

        return Prediction(
            fixture_id=match.fixture_id,
            model=self.name,
            probs=probs,
            computed_at_utc=datetime.now(timezone.utc),
            version=self.version,
            status=PredictionStatus.OK,
        )
