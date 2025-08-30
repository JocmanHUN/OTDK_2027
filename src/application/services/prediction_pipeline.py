from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol

from src.domain.entities.match import Match
from src.domain.entities.prediction import Prediction
from src.domain.interfaces.context import ModelContext
from src.domain.interfaces.modeling import BasePredictiveModel
from src.domain.value_objects.enums import PredictionStatus
from src.domain.value_objects.ids import FixtureId, LeagueId, TeamId


class _HistorySvc(Protocol):
    def get_team_averages(
        self, team_id: int, league_id: int, season: int
    ) -> Any: ...  # pragma: no cover
    def simple_poisson_means(
        self, home: Any, away: Any
    ) -> tuple[float, float]: ...  # pragma: no cover


class _FixturesSvc(Protocol):
    def get_daily_fixtures(
        self,
        for_date: str,
        *,
        league_id: int | None = None,
        season: int | None = None,
        tz_name: str = "Europe/Budapest",
    ) -> list[Mapping[str, Any]]: ...  # pragma: no cover


@dataclass
class ContextBuilder:
    history: _HistorySvc

    def build_from_meta(
        self,
        fixture_id: int,
        league_id: int,
        season: int,
        home_team_id: int,
        away_team_id: int,
        *,
        home_advantage: float = 100.0,
        init_elo: float = 1500.0,
    ) -> ModelContext:
        # Team averages (home/away) → simple Poisson means
        home_avg = self.history.get_team_averages(home_team_id, league_id, season)
        away_avg = self.history.get_team_averages(away_team_id, league_id, season)
        mu_home, mu_away = self.history.simple_poisson_means(home_avg, away_avg)

        return ModelContext(
            fixture_id=FixtureId(int(fixture_id)),
            league_id=LeagueId(int(league_id)),
            season=int(season),
            home_team_id=TeamId(int(home_team_id)),
            away_team_id=TeamId(int(away_team_id)),
            home_goal_rate=float(mu_home),
            away_goal_rate=float(mu_away),
            elo_home=float(init_elo),
            elo_away=float(init_elo),
            home_advantage=float(home_advantage),
            features=None,
        )


@dataclass
class PredictionAggregatorImpl:
    def run_all(
        self, models: list[BasePredictiveModel], match: Match, ctx: ModelContext
    ) -> list[Prediction]:
        out: list[Prediction] = []
        for m in models:
            # If minimal inputs are lacking, return SKIPPED
            if not ctx.has_minimal_inputs_for(m.name):
                out.append(
                    Prediction(
                        fixture_id=match.fixture_id,
                        model=m.name,
                        probs=None,
                        computed_at_utc=datetime.now(timezone.utc),
                        version=getattr(m, "version", "0"),
                        status=PredictionStatus.SKIPPED,
                        skip_reason=f"Missing minimal inputs for {m.name}",
                    )
                )
                continue
            try:
                out.append(m.predict(match, ctx))
            except Exception as exc:  # pragma: no cover - model bugs should not crash pipeline
                out.append(
                    Prediction(
                        fixture_id=match.fixture_id,
                        model=m.name,
                        probs=None,
                        computed_at_utc=datetime.now(timezone.utc),
                        version=getattr(m, "version", "0"),
                        status=PredictionStatus.SKIPPED,
                        skip_reason=f"Model error: {exc}",
                    )
                )
        return out
