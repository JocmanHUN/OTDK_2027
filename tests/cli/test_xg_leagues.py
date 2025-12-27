# mypy: ignore-errors

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import pytest

from src.application.services.leagues_service import LeagueSeason
from src.cli import xg_leagues
from src.cli.xg_leagues import (
    _FilterConfig,
    _fixture_has_xg,
    _iter_recent_fixture_ids,
    find_leagues_with_xg,
)


class _HistoryFake:
    def __init__(
        self,
        fixtures: dict[tuple[int, int], list[dict[str, Any]]],
        stats: dict[int, dict[int, dict[str, float]]],
    ) -> None:
        self._fixtures = fixtures
        self._stats = stats

    def get_league_finished_fixtures(self, league_id: int, season: int) -> list[Mapping[str, Any]]:
        return list(self._fixtures.get((int(league_id), int(season)), []))

    def get_fixture_statistics(self, fixture_id: int) -> dict[int, dict[str, float]]:
        return self._stats.get(int(fixture_id), {})


def _fixture_row(fixture_id: int, minutes_offset: int) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "date_utc": datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minutes_offset),
    }


def test_find_leagues_with_xg_filters_successfully() -> None:
    leagues: list[LeagueSeason] = [
        {
            "league_id": 1,
            "league_name": "Alpha",
            "country_name": "A",
            "season_year": 2025,
            "has_odds": True,
            "has_stats": True,
        },
        {
            "league_id": 2,
            "league_name": "Beta",
            "country_name": "B",
            "season_year": 2025,
            "has_odds": True,
            "has_stats": True,
        },
    ]
    fixtures = {
        (1, 2025): [
            _fixture_row(101, 0),
            _fixture_row(102, 10),
            _fixture_row(103, 20),
            _fixture_row(104, 30),
        ],
        (2, 2025): [_fixture_row(201, 0), _fixture_row(202, 10), _fixture_row(203, 20)],
    }
    stats: dict[int, dict[int, dict[str, float]]] = {
        101: {10: {"expected goals": 1.2}},
        102: {11: {"xg": 0.8}},
        103: {12: {"shots": 5.0}},
        104: {13: {"expected goals": 0.0}},
        201: {21: {"shots": 7.0}},
        202: {22: {"possession": 55.0}},
        203: {23: {"xg": 0.0}},
    }
    history = _HistoryFake(fixtures, stats)

    cfg = _FilterConfig(min_checks=3, min_success=2, max_probe=4, stat_delay=0)
    result = find_leagues_with_xg(leagues, history, cfg)
    assert len(result) == 1
    assert result[0]["league_id"] == 1


def test_league_dropped_if_not_enough_checks_available() -> None:
    leagues: list[LeagueSeason] = [
        {
            "league_id": 3,
            "league_name": "Gamma",
            "country_name": "C",
            "season_year": 2025,
            "has_odds": True,
            "has_stats": True,
        }
    ]
    fixtures = {(3, 2025): [_fixture_row(301, 0), _fixture_row(302, 10)]}
    stats: dict[int, dict[int, dict[str, float]]] = {
        301: {31: {"expected goals": 0.5}},
        302: {32: {"xg": 0.7}},
    }
    history = _HistoryFake(fixtures, stats)

    cfg = _FilterConfig(min_checks=3, min_success=2, max_probe=5, stat_delay=0)
    result = find_leagues_with_xg(leagues, history, cfg)
    assert result == []


def test_helpers_and_clamp() -> None:
    stats = {1: {"expGoals": None}, 2: {"expected goals": 0.0}}
    assert _fixture_has_xg(stats)
    assert _fixture_has_xg({1: {"shots": 1.0}}) is False

    fixtures = [
        {"fixture_id": 1, "date_utc": datetime(2024, 1, 1, tzinfo=timezone.utc)},
        {"fixture_id": 2, "date_utc": datetime(2024, 1, 2, tzinfo=timezone.utc)},
    ]
    assert list(_iter_recent_fixture_ids(fixtures)) == [2, 1]

    cfg = _FilterConfig(min_checks=0, min_success=0, max_probe=0, stat_delay=-1).clamp()
    assert cfg.min_checks == 1 and cfg.min_success == 1 and cfg.max_probe == 1
    assert cfg.stat_delay == 0.0


def test_xg_leagues_main_no_results(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # Force no qualified leagues
    class _LSvc:
        def get_current_leagues(self):
            return [{"league_id": 1, "season_year": 2024}]

    class _HSvc:
        def get_league_finished_fixtures(self, league_id: int, season: int):
            return [{"fixture_id": 10, "date_utc": datetime.now(timezone.utc)}]

        def get_fixture_statistics(self, fixture_id: int):
            return {1: {"shots": 5.0}}

    monkeypatch.setattr(xg_leagues, "LeaguesService", lambda: _LSvc())
    monkeypatch.setattr(xg_leagues, "HistoryService", lambda: _HSvc())
    rc = xg_leagues.main(["--output", str(tmp_path / "out.json"), "--stat-delay", "0"])
    assert rc == 1
    assert "No leagues with reliable xG statistics" in capsys.readouterr().out


def test_xg_leagues_main_writes_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class _LSvc:
        def get_current_leagues(self):
            return [{"league_id": 1, "season_year": 2024, "league_name": "L"}]

    class _HSvc:
        def __init__(self) -> None:
            self.calls = 0

        def get_league_finished_fixtures(self, league_id: int, season: int):
            return [{"fixture_id": 10, "date_utc": datetime(2024, 1, 1, tzinfo=timezone.utc)}]

        def get_fixture_statistics(self, fixture_id: int):
            self.calls += 1
            return {1: {"xg": 1.0}}

    monkeypatch.setattr(xg_leagues, "LeaguesService", lambda: _LSvc())
    monkeypatch.setattr(xg_leagues, "HistoryService", lambda: _HSvc())
    monkeypatch.setattr(xg_leagues.time, "sleep", lambda s: None)

    out_file = tmp_path / "xg.json"
    rc = xg_leagues.main(
        ["--output", str(out_file), "--min-checks", "1", "--min-success", "1", "--stat-delay", "0"]
    )
    assert rc == 0
    data = out_file.read_text(encoding="utf-8")
    assert '"league_id": 1' in data
    assert "Wrote 1 leagues" in capsys.readouterr().out
