from __future__ import annotations

from typing import Any, List, Mapping

import pytest


class _FakeFixturesService:
    def __init__(self, rows: List[dict[str, Any]]):
        self.rows = rows
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def get_daily_fixtures(
        self,
        for_date: str,
        *,
        league_id: int | None = None,
        season: int | None = None,
        tz_name: str = "Europe/Budapest",
        debug: bool = False,
    ) -> List[dict[str, Any]]:
        self.calls.append((for_date, {"league": league_id, "season": season, "tz": tz_name}))
        return self.rows


@pytest.fixture
def sample_rows() -> List[dict[str, Any]]:
    return [
        {
            "fixture_id": 1,
            "league_id": 39,
            "season": 2025,
            "date_utc": "2025-03-10T16:00:00+00:00",
            "home_id": 10,
            "away_id": 20,
            "home_name": "A",
            "away_name": "B",
            "status": "NS",
        }
    ]


def test_fixtures_cli_with_filters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sample_rows: List[dict[str, Any]],
) -> None:
    import src.cli.fixtures as fixtures_cli

    fake = _FakeFixturesService(sample_rows)
    monkeypatch.setattr(fixtures_cli, "FixturesService", lambda: fake)

    rc = fixtures_cli.main(
        ["--date", "2025-03-10", "--league", "39", "--season", "2025"]
    )  # default timezone
    assert rc == 0
    out = capsys.readouterr().out
    assert "A vs B" in out and "league=39" in out and "season=2025" in out
    assert fake.calls == [("2025-03-10", {"league": 39, "season": 2025, "tz": "Europe/Budapest"})]


def test_fixtures_cli_empty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import src.cli.fixtures as fixtures_cli

    fake = _FakeFixturesService([])
    monkeypatch.setattr(fixtures_cli, "FixturesService", lambda: fake)

    rc = fixtures_cli.main(["--date", "2025-03-10"])  # no filters
    assert rc == 0
    out = capsys.readouterr().out
    assert "No fixtures found." in out
