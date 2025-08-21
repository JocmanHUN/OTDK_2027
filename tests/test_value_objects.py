from decimal import Decimal

import pytest

from src.domain.value_objects.enums import Outcome
from src.domain.value_objects.ids import LeagueId
from src.domain.value_objects.money import Money
from src.domain.value_objects.probability_triplet import ProbabilityTriplet


def test_probability_triplet_validation() -> None:
    with pytest.raises(ValueError):
        ProbabilityTriplet(home=0.5, draw=0.4, away=0.2)


def test_probability_triplet_normalization() -> None:
    triplet = ProbabilityTriplet(home=0.5, draw=0.3, away=0.2)
    normalized = triplet.normalized()
    assert pytest.approx(normalized.home + normalized.draw + normalized.away, 1e-9) == 1


def test_probability_triplet_from_odds() -> None:
    odds = [2, 3, 4]
    triplet = ProbabilityTriplet.from_odds(odds)
    assert pytest.approx(triplet.home + triplet.draw + triplet.away, 1e-9) == 1


def test_money_operations() -> None:
    m1 = Money(amount=Decimal("10"), currency="EUR")
    m2 = Money(amount=Decimal("5"), currency="EUR")
    assert (m1 + m2).amount == Decimal("15.00")
    assert (m1 - m2).amount == Decimal("5.00")
    assert (m2 * 2).amount == Decimal("10.00")
    assert m1.ratio_to(m2) == Decimal("2")


def test_money_currency_mismatch() -> None:
    m1 = Money(amount=Decimal("10"), currency="EUR")
    m2 = Money(amount=Decimal("5"), currency="USD")
    with pytest.raises(ValueError):
        _ = m1 + m2


def test_enums_and_ids() -> None:
    league_id: LeagueId = LeagueId(10)
    assert isinstance(league_id, int)
    assert Outcome.HOME.value == "HOME"
