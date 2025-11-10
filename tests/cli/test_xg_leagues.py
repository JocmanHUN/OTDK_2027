from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from src.application.services.leagues_service import LeagueSeason
from src.cli.xg_leagues import _FilterConfig, find_leagues_with_xg


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
