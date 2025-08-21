from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal

from src.domain.entities.match import Match
from src.domain.entities.odds import Odds
from src.domain.entities.prediction import Prediction
from src.domain.entities.strategy_result import StrategyResult
from src.domain.interfaces.context import ModelContext
from src.domain.interfaces.modeling import BasePredictiveModel
from src.domain.interfaces.strategies import BaseStrategy
from src.domain.value_objects.enums import (
    MatchStatus,
    ModelName,
    Outcome,
    PredictionStatus,
    StrategyName,
)
from src.domain.value_objects.ids import BookmakerId, FixtureId, LeagueId
from src.domain.value_objects.money import Money
from src.domain.value_objects.probability_triplet import ProbabilityTriplet


class DummyAlwaysHome(BasePredictiveModel):
    name = ModelName.POISSON
    version = "0"

    def predict(self, match: Match, ctx: ModelContext) -> Prediction:
        probs = ProbabilityTriplet(home=1.0, draw=0.0, away=0.0)
        return Prediction(
            fixture_id=match.fixture_id,
            model=self.name,
            probs=probs,
            computed_at_utc=datetime.now(timezone.utc),
            version=self.version,
            status=PredictionStatus.OK,
        )


class Flat1Unit(BaseStrategy):
    name = StrategyName.FLAT
    version = "1.0"

    def run(
        self,
        *,
        bankroll: Money,
        odds: Odds,
        prediction: Prediction,
        outcome: Outcome | None = None,
        step_index: int = 0,
    ) -> StrategyResult:
        target = Outcome.HOME
        result: Literal["WIN", "LOSE", "VOID"] | None
        if outcome is None:
            stake = Money(amount=Decimal("0"), currency=bankroll.currency)
            result = None
            profit = Money(amount=Decimal("0"), currency=bankroll.currency)
            bankroll_after = bankroll
        elif outcome == target:
            stake = Money(amount=Decimal("1"), currency=bankroll.currency)
            result = "WIN"
            profit = stake * (odds.home - Decimal("1"))
            bankroll_after = bankroll + profit
        else:
            stake = Money(amount=Decimal("1"), currency=bankroll.currency)
            result = "LOSE"
            profit = stake * Decimal("-1")
            bankroll_after = bankroll + profit
        return StrategyResult(
            fixture_id=odds.fixture_id,
            strategy=self.name,
            model=prediction.model,
            outcome=target,
            stake=stake,
            odds=odds.home,
            result=result,
            profit=profit,
            bankroll_after=bankroll_after,
            step_index=step_index,
        )


def test_dummy_model_and_strategy() -> None:
    match = Match(
        fixture_id=FixtureId(1),
        league_id=LeagueId(1),
        season=2024,
        kickoff_utc=datetime.now(timezone.utc),
        home_name="A",
        away_name="B",
        status=MatchStatus.SCHEDULED,
    )
    ctx = ModelContext(fixture_id=match.fixture_id, league_id=match.league_id, season=match.season)
    model = DummyAlwaysHome()
    pred = model.predict(match, ctx)
    assert pred.status == PredictionStatus.OK
    odds = Odds(
        fixture_id=match.fixture_id,
        bookmaker_id=BookmakerId(1),
        collected_at_utc=datetime.now(timezone.utc),
        home=Decimal("2.0"),
        draw=Decimal("3.0"),
        away=Decimal("4.0"),
    )
    strat = Flat1Unit()
    bankroll = Money(amount=Decimal("10"), currency="HUF")
    result = strat.run(bankroll=bankroll, odds=odds, prediction=pred, outcome=Outcome.HOME)
    assert isinstance(result, StrategyResult)
    assert result.bankroll_after.amount > bankroll.amount
