from __future__ import annotations

import os
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
    def get_recent_team_scores(
        self, team_id: int, league_id: int, season: int, last: int, *, only_finished: bool = True
    ) -> list[dict[str, Any]]: ...  # pragma: no cover
    def league_goal_means(
        self, league_id: int, season: int
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
    compute_features: bool = True
    features_last: int = int(os.getenv("FEATURES_LAST", "10"))

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

        # Override with cheap recent-scores Poisson (no per-fixture stats), with league shrinkage + floors
        try:
            last_n = int(self.features_last)
        except Exception:
            last_n = 10
        sample_min = int(os.getenv("POISSON_SAMPLE_MIN", "5"))
        floor_home = float(os.getenv("POISSON_FLOOR_HOME", "0.2"))
        floor_away = float(os.getenv("POISSON_FLOOR_AWAY", "0.2"))
        prior_denom = float(os.getenv("POISSON_PRIOR_WEIGHT_DENOM", "20"))
        try:
            home_rows = self.history.get_recent_team_scores(
                home_team_id, league_id, season, last_n, only_finished=True
            )
        except Exception:
            home_rows = []
        try:
            away_rows = self.history.get_recent_team_scores(
                away_team_id, league_id, season, last_n, only_finished=True
            )
        except Exception:
            away_rows = []

        def _mean(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        h_home = [float(r.get("goals_for", 0.0)) for r in home_rows if r.get("home_away") == "H"]
        h_all = [float(r.get("goals_for", 0.0)) for r in home_rows]
        mu_home_recent = _mean(h_home) if len(h_home) >= max(1, sample_min // 2) else _mean(h_all)
        a_away = [float(r.get("goals_for", 0.0)) for r in away_rows if r.get("home_away") == "A"]
        a_all = [float(r.get("goals_for", 0.0)) for r in away_rows]
        mu_away_recent = _mean(a_away) if len(a_away) >= max(1, sample_min // 2) else _mean(a_all)
        try:
            mu_h_league, mu_a_league = self.history.league_goal_means(league_id, season)
        except Exception:
            mu_h_league, mu_a_league = (1.35, 1.15)
        n = len(home_rows) + len(away_rows)
        w = min(1.0, (n / prior_denom) if prior_denom > 0 else 1.0)
        mu_home = max(floor_home, w * mu_home_recent + (1.0 - w) * mu_h_league)
        mu_away = max(floor_away, w * mu_away_recent + (1.0 - w) * mu_a_league)

        # Build feature diffs from recent stats for logistic regression models
        feats = None
        if self.compute_features:
            try:
                from src.application.services.features_service import FeaturesService as _FeatSvc

                feats = _FeatSvc(self.history).build_features(
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    league_id=league_id,
                    season=season,
                    last=int(self.features_last),
                )
            except Exception:
                feats = None

        return ModelContext(
            fixture_id=FixtureId(int(fixture_id)),
            league_id=LeagueId(int(league_id)),
            season=int(season),
            home_team_id=TeamId(int(home_team_id)),
            away_team_id=TeamId(int(away_team_id)),
            home_goal_rate=float(mu_home),
            away_goal_rate=float(mu_away),
            home_advantage=float(home_advantage),
            features=feats,
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
