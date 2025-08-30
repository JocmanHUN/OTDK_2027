from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Sequence

import pytest


class _FakePipe:
    def __init__(
        self,
        leagues: Sequence[Mapping[str, Any]],
        fixtures: Sequence[Mapping[str, Any]],
        with_odds_ids: Sequence[int] = (),
    ) -> None:  # noqa: D401
        self._leagues = leagues
        self._fixtures = fixtures
        self._with_odds_ids = set(with_odds_ids)

    def list_supported_leagues(self) -> List[Mapping[str, Any]]:
        return list(self._leagues)

    def list_daily_matches(
        self,
        for_date: str,
        leagues: Iterable[Mapping[str, Any]],
        *,
        tz_name: str = "Europe/Budapest",
    ) -> List[Mapping[str, Any]]:  # noqa: D401
        return list(self._fixtures)

    def filter_matches_with_1x2_odds(
        self, fixtures: Iterable[Mapping[str, Any]]
    ) -> List[Mapping[str, Any]]:
        return [f for f in fixtures if int(f["fixture_id"]) in self._with_odds_ids]


def test_select_cli_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import src.cli.select as select_cli

    leagues: Sequence[Mapping[str, Any]] = [{"league_id": 39, "season_year": 2025}]
    fixtures: Sequence[Mapping[str, Any]] = [
        {"fixture_id": 1001, "league_id": 39, "season_year": 2025, "home_id": 42, "away_id": 40},
        {"fixture_id": 1002, "league_id": 39, "season_year": 2025, "home_id": 50, "away_id": 49},
    ]
    fake = _FakePipe(leagues, fixtures, with_odds_ids=[1001])
    monkeypatch.setattr(select_cli.SelectionPipeline, "default", staticmethod(lambda: fake))

    rc = select_cli.main(["--date", "2025-08-31", "--with-odds"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "leagues=1" in out and "fixtures=2" in out and "fixtures_with_odds=1" in out


def test_select_cli_print(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import src.cli.select as select_cli

    leagues: Sequence[Mapping[str, Any]] = [{"league_id": 39, "season_year": 2025}]
    fixtures: Sequence[Mapping[str, Any]] = [
        {"fixture_id": 2001, "league_id": 39, "season_year": 2025, "home_id": 1, "away_id": 2}
    ]
    fake = _FakePipe(leagues, fixtures, with_odds_ids=[2001])
    monkeypatch.setattr(select_cli.SelectionPipeline, "default", staticmethod(lambda: fake))

    rc = select_cli.main(["--date", "2025-08-31", "--with-odds", "--print-fixtures"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fixture=2001" in out and "league=39" in out
