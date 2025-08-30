from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import List

import pytest

from src.domain.entities.odds import Odds
from src.domain.value_objects.ids import BookmakerId, FixtureId


class _FakeOddsService:
    def __init__(self, items: List[Odds]):
        self.items = items
        self.calls: list[int] = []

    def get_fixture_odds(self, fixture_id: int) -> List[Odds]:
        self.calls.append(fixture_id)
        return self.items


def _odds(fid: int, bid: int, h: str, d: str, a: str) -> Odds:
    return Odds(
        fixture_id=FixtureId(fid),
        bookmaker_id=BookmakerId(bid),
        collected_at_utc=datetime.now(timezone.utc),
        home=Decimal(h),
        draw=Decimal(d),
        away=Decimal(a),
    )


@pytest.fixture
def sample_odds() -> List[Odds]:
    return [
        _odds(100, 10, "2.00", "3.50", "4.00"),
        _odds(100, 20, "1.90", "3.60", "4.20"),
    ]


def test_odds_cli_list(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], sample_odds: List[Odds]
) -> None:
    import src.cli.odds as odds_cli

    fake = _FakeOddsService(sample_odds)
    monkeypatch.setattr(odds_cli, "OddsService", lambda: fake)

    rc = odds_cli.main(["--fixture", "100"])  # list all
    assert rc == 0
    out = capsys.readouterr().out
    assert "bookmaker=10" in out and "bookmaker=20" in out
    assert "H=2.00 D=3.50 A=4.00" in out


def test_odds_cli_best(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], sample_odds: List[Odds]
) -> None:
    import src.cli.odds as odds_cli

    fake = _FakeOddsService(sample_odds)
    monkeypatch.setattr(odds_cli, "OddsService", lambda: fake)

    rc = odds_cli.main(["--fixture", "100", "--best"])
    assert rc == 0
    out = capsys.readouterr().out
    # Best HOME should be bookmaker 10 at 2.00; best AWAY bookmaker 20 at 4.20
    assert "Best HOME:" in out and "bookmaker=10" in out
    assert "Best AWAY:" in out and "odd=4.20" in out
