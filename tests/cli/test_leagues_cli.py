from __future__ import annotations

from typing import Any, List

import pytest


class _FakeService:
    def __init__(self, rows: List[dict[str, Any]]):
        self.rows = rows
        self.calls: List[str] = []

    def get_current_leagues(self) -> List[dict[str, Any]]:
        self.calls.append("current")
        return self.rows

    def get_leagues_for_season(self, year: int) -> List[dict[str, Any]]:
        self.calls.append(f"season:{year}")
        return self.rows


@pytest.fixture
def sample_rows() -> List[dict[str, Any]]:
    return [
        {
            "league_id": 1,
            "league_name": "Premier League",
            "country_name": "England",
            "season_year": 2024,
            "has_odds": True,
            "has_stats": True,
        }
    ]


def test_cli_current(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sample_rows: List[dict[str, Any]],
) -> None:

    # Patch LeaguesService constructor to return our fake
    import src.cli.leagues as leagues_cli

    fake = _FakeService(sample_rows)
    monkeypatch.setattr(leagues_cli, "LeaguesService", lambda: fake)

    rc = leagues_cli.main(["--current"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Premier League" in out
    assert fake.calls == ["current"]


def test_cli_season(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sample_rows: List[dict[str, Any]],
) -> None:
    import src.cli.leagues as leagues_cli

    fake = _FakeService(sample_rows)
    monkeypatch.setattr(leagues_cli, "LeaguesService", lambda: fake)

    rc = leagues_cli.main(["--season", "2023"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2024" in out  # the sample row has season_year=2024 and should be printed
    assert fake.calls == ["season:2023"]
