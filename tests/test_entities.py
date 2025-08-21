from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.domain.entities.match import Match
from src.domain.entities.odds import Odds, best_of
from src.domain.entities.prediction import Prediction
from src.domain.entities.strategy_result import StrategyResult
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


def test_match_requires_utc_and_scores() -> None:
    with pytest.raises(ValueError):
        Match(
            fixture_id=FixtureId(1),
            league_id=LeagueId(2),
            season=2024,
            kickoff_utc=datetime(2024, 5, 1),
            home_name="A",
            away_name="B",
            status=MatchStatus.SCHEDULED,
        )
    with pytest.raises(ValueError):
        Match(
            fixture_id=FixtureId(1),
            league_id=LeagueId(2),
            season=2024,
            kickoff_utc=datetime(2024, 5, 1, tzinfo=timezone.utc),
            home_name="A",
            away_name="B",
            status=MatchStatus.FINISHED,
        )
    m = Match(
        fixture_id=FixtureId(1),
        league_id=LeagueId(2),
        season=2024,
        kickoff_utc=datetime(2024, 5, 1, tzinfo=timezone.utc),
        home_name="A",
        away_name="B",
        status=MatchStatus.FINISHED,
        ft_home_goals=1,
        ft_away_goals=0,
    )
    assert m.ft_home_goals == 1


def test_odds_validation_and_best_of() -> None:
    with pytest.raises(ValueError):
        Odds(
            fixture_id=FixtureId(1),
            bookmaker_id=BookmakerId(1),
            collected_at_utc=datetime.now(timezone.utc),
            home=Decimal("1.0"),
            draw=Decimal("2.0"),
            away=Decimal("3.0"),
        )
    o1 = Odds(
        fixture_id=FixtureId(1),
        bookmaker_id=BookmakerId(1),
        collected_at_utc=datetime.now(timezone.utc),
        home=Decimal("2.0"),
        draw=Decimal("3.0"),
        away=Decimal("4.0"),
    )
    o2 = Odds(
        fixture_id=FixtureId(1),
        bookmaker_id=BookmakerId(2),
        collected_at_utc=datetime.now(timezone.utc),
        home=Decimal("2.5"),
        draw=Decimal("2.5"),
        away=Decimal("4.5"),
    )
    probs = o1.implied_probabilities()
    assert pytest.approx(probs.home + probs.draw + probs.away, 1e-9) == 1
    best = best_of([o1, o2])
    assert best[Outcome.HOME] == (BookmakerId(2), Decimal("2.5"))
    assert best[Outcome.AWAY] == (BookmakerId(2), Decimal("4.5"))


def test_prediction_ev_and_best_ev() -> None:
    probs = ProbabilityTriplet(home=0.5, draw=0.3, away=0.2)
    pred = Prediction(
        fixture_id=FixtureId(1),
        model=ModelName.POISSON,
        probs=probs,
        computed_at_utc=datetime.now(timezone.utc),
        version="1.0",
        status=PredictionStatus.OK,
    )
    odds = Odds(
        fixture_id=FixtureId(1),
        bookmaker_id=BookmakerId(1),
        collected_at_utc=datetime.now(timezone.utc),
        home=Decimal("2.0"),
        draw=Decimal("3.0"),
        away=Decimal("4.0"),
    )
    ev_home = pred.ev(odds, Outcome.HOME)
    assert ev_home == Decimal("0")
    o2 = Odds(
        fixture_id=FixtureId(1),
        bookmaker_id=BookmakerId(2),
        collected_at_utc=datetime.now(timezone.utc),
        home=Decimal("2.5"),
        draw=Decimal("2.5"),
        away=Decimal("4.5"),
    )
    outcome, bookmaker, value = pred.best_ev([odds, o2])
    assert outcome == Outcome.HOME
    assert bookmaker == BookmakerId(2)
    assert value == pred.ev(o2, Outcome.HOME)


def test_prediction_status_requirements() -> None:
    with pytest.raises(ValueError):
        Prediction(
            fixture_id=FixtureId(1),
            model=ModelName.POISSON,
            computed_at_utc=datetime.now(timezone.utc),
            version="1.0",
            status=PredictionStatus.OK,
        )
    with pytest.raises(ValueError):
        Prediction(
            fixture_id=FixtureId(1),
            model=ModelName.POISSON,
            computed_at_utc=datetime.now(timezone.utc),
            version="1.0",
            status=PredictionStatus.SKIPPED,
        )


def test_strategy_result_creation() -> None:
    res = StrategyResult(
        fixture_id=FixtureId(1),
        strategy=StrategyName.FLAT,
        model=ModelName.POISSON,
        outcome=Outcome.HOME,
        stake=Money(amount=Decimal("10"), currency="EUR"),
        odds=Decimal("2.0"),
        profit=Money(amount=Decimal("10"), currency="EUR"),
        bankroll_after=Money(amount=Decimal("110"), currency="EUR"),
        step_index=0,
    )
    assert res.bankroll_after.amount == Decimal("110.00")
