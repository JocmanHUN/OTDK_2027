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
    """Asymmetric Veto modell a megadott képlet szerint.

    Paraméterek
    - `history`: adatforrás `get_recent_team_stats(...)` metódussal (teszteléshez cserélhető).
    - `last_n`: legfeljebb ennyi legutóbbi meccs számít (<=0 → üres történelem).
    - `decay_factor`: exponenciális súly a múltra, [0,1] közé szorítva, 0 pedig 1-nek számít.

    Kombináció
    - P(1) = P_H(win) * (1 - P_A(win))
    - P(2) = P_A(win) * (1 - P_H(win))
    - P(X) = (P_H(draw) * n_H + P_A(draw) * n_A) / (n_H + n_A)
    Végül normalizáljuk a P(1), P(X), P(2) hármast (belsőleg 0-1 tartományban tartva).
    """

    name: ClassVar[ModelName] = ModelName.VETO
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

        home_rows = self.history.get_recent_team_stats(
            int(ctx.home_team_id), int(ctx.league_id), int(ctx.season), self.last_n
        )
        away_rows = self.history.get_recent_team_stats(
            int(ctx.away_team_id), int(ctx.league_id), int(ctx.season), self.last_n
        )

        hW, hD, _ = _form_distribution(home_rows, self.decay_factor)
        aW, aD, _ = _form_distribution(away_rows, self.decay_factor)

        n_home = len(home_rows)
        n_away = len(away_rows)
        n_total = n_home + n_away

        p1_raw = hW * (1.0 - aW)
        p2_raw = aW * (1.0 - hW)

        if n_total > 0:
            px_raw = (hD * n_home + aD * n_away) / n_total
        else:
            px_raw = 1.0 / 3.0  # nincs adat, maradjon semleges

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
