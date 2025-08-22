
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.domain.entities.odds import Odds, best_of
from src.domain.entities.prediction import Prediction
from src.domain.value_objects.enums import ModelName, Outcome, PredictionStatus
from src.domain.value_objects.ids import BookmakerId, FixtureId
from src.domain.value_objects.money import Money
from src.domain.value_objects.probability_triplet import ProbabilityTriplet
from src import logging_config

def test_odds_timezone_and_best_of_branch() -> None:
    with pytest.raises(ValueError):
        Odds(
            fixture_id=FixtureId(1),
            bookmaker_id=BookmakerId(1),
            collected_at_utc=datetime.now(),  # naive datetime
            home=Decimal("2.0"),
            draw=Decimal("3.0"),
            away=Decimal("4.0"),
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
        home=Decimal("1.5"),
        draw=Decimal("2.5"),
        away=Decimal("3.5"),
    )
    best = best_of([o1, o2])
    assert best[Outcome.HOME] == (o1.bookmaker_id, o1.home)
    assert best[Outcome.DRAW] == (o1.bookmaker_id, o1.draw)
    assert best[Outcome.AWAY] == (o1.bookmaker_id, o1.away)

def test_prediction_timezone_and_missing_probs() -> None:
    with pytest.raises(ValueError):
        Prediction(
            fixture_id=FixtureId(1),
            model=ModelName.POISSON,
            probs=ProbabilityTriplet(home=0.4, draw=0.3, away=0.3),
            computed_at_utc=datetime.now(),  # naive datetime
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
    pred = Prediction(
        fixture_id=FixtureId(1),
        model=ModelName.POISSON,
        computed_at_utc=datetime.now(timezone.utc),
        version="1.0",
        status=PredictionStatus.SKIPPED,
        skip_reason="n/a",
    )
    with pytest.raises(ValueError):
        pred.ev(odds, Outcome.HOME)

def test_money_and_probability_triplet_edge_cases() -> None:
    m1 = Money(amount=10, currency="EUR")
    assert m1.amount == Decimal("10.00")
    m_zero = Money(amount=0, currency="EUR")
    with pytest.raises(ZeroDivisionError):
        m1.ratio_to(m_zero)

    with pytest.raises(ValueError):
        ProbabilityTriplet(home=1.2, draw=0.0, away=-0.2)

    triplet_zero = ProbabilityTriplet.model_construct(home=0.0, draw=0.0, away=0.0)
    with pytest.raises(ValueError):
        triplet_zero.normalized()

    triplet = ProbabilityTriplet.model_construct(home=0.4, draw=0.2, away=0.2)
    normalized = triplet.normalized()
    assert pytest.approx(normalized.home + normalized.draw + normalized.away, 1e-9) == 1
    assert normalized != triplet
    triplet_valid = ProbabilityTriplet(home=0.4, draw=0.3, away=0.3)
    assert triplet_valid.as_dict()["home"] == pytest.approx(0.4)

    with pytest.raises(ValueError):
        ProbabilityTriplet.from_odds([2, 3])

    with pytest.raises(ValueError):
        ProbabilityTriplet.from_odds([2, 3, 4], remove_overround=False)

def test_logging_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(logging_config, "LOG_FILE", tmp_path / "app.log")
    logger = logging.getLogger(logging_config.LOG_NAME)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.propagate = True

    logger = logging_config.get_logger()
    assert logging_config.LOG_FILE.exists()
    logger2 = logging_config.get_logger()
    assert logger2 is logger

    formatter = logging_config.JsonFormatter()
    record = logger.makeRecord(logger.name, logging.INFO, __file__, 0, "msg", (), None)
    record.request_id = "req1"
    record.user = "bob"
    output = formatter.format(record)
    data = json.loads(output)
    assert data["request_id"] == "req1"
    assert data["extra"]["user"] == "bob"

    stats = logging_config.CacheStats()
    stats.record_hit()
    stats.record_miss()
    assert stats.hit_rate == 50.0
    stats.log_hit_rate()
