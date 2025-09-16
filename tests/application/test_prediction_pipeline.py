from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.application.services.prediction_pipeline import ContextBuilder, PredictionAggregatorImpl
from src.domain.entities.match import Match
from src.domain.entities.prediction import Prediction
from src.domain.interfaces.context import ModelContext
from src.domain.interfaces.modeling import BasePredictiveModel
from src.domain.value_objects.enums import MatchStatus, ModelName, PredictionStatus
from src.domain.value_objects.ids import FixtureId, LeagueId
from src.domain.value_objects.probability_triplet import ProbabilityTriplet


class _HistFake:
    class _TA:
        def __init__(self, gf_home: float, gf_away: float, ga_home: float, ga_away: float) -> None:
            self.goals_for_home_avg = gf_home
            self.goals_for_away_avg = gf_away
            self.goals_against_home_avg = ga_home
            self.goals_against_away_avg = ga_away

    def get_team_averages(self, team_id: int, league_id: int, season: int) -> Any:
        # Return stable values, independent of ids
        return self._TA(2.0, 1.0, 1.1, 1.3)

    def simple_poisson_means(self, home: Any, away: Any) -> tuple[float, float]:
        # mean(home.GF_home, away.GA_away), mean(away.GF_away, home.GA_home)
        mu_home = (home.goals_for_home_avg + away.goals_against_away_avg) / 2.0
        mu_away = (away.goals_for_away_avg + home.goals_against_home_avg) / 2.0
        return (mu_home, mu_away)

    # New lightweight interfaces used by ContextBuilder
    def get_recent_team_scores(
        self, team_id: int, league_id: int, season: int, last: int, *, only_finished: bool = True
    ) -> list[dict[str, Any]]:
        # Provide a minimal set so Poisson recent-scores path has data
        out: list[dict[str, Any]] = []
        for i in range(max(1, last)):
            out.append(
                {
                    "fixture_id": 1000 + i,
                    "date_utc": datetime.now(timezone.utc),
                    "home_away": "H" if i % 2 == 0 else "A",
                    "goals_for": 1 + (i % 2),
                    "goals_against": (i % 2),
                }
            )
        return out[:last]

    def league_goal_means(self, league_id: int, season: int) -> tuple[float, float]:
        return (1.35, 1.15)


class _PoissonDummy(BasePredictiveModel):
    name = ModelName.POISSON
    version = "1"

    def predict(self, match: Match, ctx: ModelContext) -> Prediction:
        if not ctx.has_minimal_inputs_for(self.name):
            return Prediction(
                fixture_id=match.fixture_id,
                model=self.name,
                probs=None,
                computed_at_utc=datetime.now(timezone.utc),
                version=self.version,
                status=PredictionStatus.SKIPPED,
                skip_reason="missing inputs",
            )
        # Dumb symmetric probs based on mu ratio
        assert ctx.home_goal_rate is not None and ctx.away_goal_rate is not None
        total = float(ctx.home_goal_rate + ctx.away_goal_rate)
        p_home = float(ctx.home_goal_rate) / total
        p_away = float(ctx.away_goal_rate) / total
        probs = ProbabilityTriplet(home=p_home, draw=0.0, away=p_away)
        return Prediction(
            fixture_id=match.fixture_id,
            model=self.name,
            probs=probs,
            computed_at_utc=datetime.now(timezone.utc),
            version=self.version,
            status=PredictionStatus.OK,
        )


def test_context_builder_and_aggregator() -> None:
    builder = ContextBuilder(history=_HistFake())
    ctx = builder.build_from_meta(
        fixture_id=10, league_id=39, season=2025, home_team_id=42, away_team_id=40
    )
    assert ctx.home_goal_rate is not None and ctx.away_goal_rate is not None

    match = Match(
        fixture_id=FixtureId(10),
        league_id=LeagueId(39),
        season=2025,
        kickoff_utc=datetime.now(timezone.utc),
        home_name="H",
        away_name="A",
        status=MatchStatus.SCHEDULED,
    )
    agg = PredictionAggregatorImpl()
    preds = agg.run_all([_PoissonDummy()], match, ctx)
    assert len(preds) == 1 and preds[0].status == PredictionStatus.OK
