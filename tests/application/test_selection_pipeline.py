# mypy: ignore-errors

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

from src.application.services import fixtures_service, leagues_service, odds_service
from src.application.services.selection_pipeline import (
    FixtureShort,
    LeagueSeason,
    SelectionPipeline,
)


class _LeaguesFake:
    def __init__(self, leagues: Sequence[Mapping[str, Any]]):
        self._leagues = leagues

    def get_current_leagues(self) -> List[Mapping[str, Any]]:
        return list(self._leagues)


class _FixturesFake:
    def __init__(self, per_league: Dict[int, List[Mapping[str, Any]]]):
        self.per_league = per_league

    def get_daily_fixtures(
        self,
        for_date: str,
        *,
        league_id: int | None = None,
        season: int | None = None,
        tz_name: str = "Europe/Budapest",
    ) -> List[Mapping[str, Any]]:
        return self.per_league.get(int(league_id or 0), [])


class _OddsFake:
    def __init__(self, with_odds: Iterable[int], raise_for: Iterable[int] = ()):  # fixture_ids
        self.with_odds = set(with_odds)
        self.raise_for = set(raise_for)

    def get_fixture_odds(self, fixture_id: int) -> List[Any]:
        if fixture_id in self.raise_for:
            raise RuntimeError("odds error")
        return [1] if fixture_id in self.with_odds else []


def test_pipeline_basic_flow() -> None:
    leagues = [
        {"league_id": 39, "season_year": 2025},
        {"league_id": 61, "season_year": 2025},
    ]
    fixtures_map: Dict[int, List[Mapping[str, Any]]] = {
        39: [
            {"fixture": {"id": 1001}, "teams": {"home": {"id": 42}, "away": {"id": 40}}},
            {"fixture": {"id": 1002}, "teams": {"home": {"id": 50}, "away": {"id": 49}}},
        ],
        61: [{"fixture": {"id": 2001}, "teams": {"home": {"id": 100}, "away": {"id": 101}}}],
    }
    odds = _OddsFake(with_odds=[1001, 2001], raise_for=[1002])

    pipe = SelectionPipeline(_LeaguesFake(leagues), _FixturesFake(fixtures_map), odds)

    supported = pipe.list_supported_leagues()
    assert supported == [
        LeagueSeason(league_id=39, season_year=2025),
        LeagueSeason(league_id=61, season_year=2025),
    ]

    fixtures = pipe.list_daily_matches("2025-08-31", supported)
    fx_ids = sorted(f["fixture_id"] for f in fixtures)
    assert fx_ids == [1001, 1002, 2001]

    filtered = pipe.filter_matches_with_1x2_odds(fixtures)
    # 1001 has odds, 1002 raises (skip), 2001 has odds
    filtered_ids = sorted(f["fixture_id"] for f in filtered)
    assert filtered_ids == [1001, 2001]


def test_selection_default_uses_stubbed_services(monkeypatch) -> None:
    class _Stub:
        def __init__(self) -> None:
            self.created = True

    monkeypatch.setattr(leagues_service, "LeaguesService", _Stub)
    monkeypatch.setattr(fixtures_service, "FixturesService", _Stub)
    monkeypatch.setattr(odds_service, "OddsService", _Stub)

    pipe = SelectionPipeline.default()
    assert isinstance(pipe.leagues_svc, _Stub)
    assert isinstance(pipe.fixtures_svc, _Stub)
    assert isinstance(pipe.odds_svc, _Stub)


def test_list_supported_leagues_skips_bad_rows() -> None:
    bad_and_good = _LeaguesFake(
        [
            {"league_id": "bad", "season_year": "oops"},
            {"league_id": 11, "season_year": 2026},
        ]
    )
    pipe = SelectionPipeline(bad_and_good, _FixturesFake({}), _OddsFake([]))
    out = pipe.list_supported_leagues()
    assert out == [LeagueSeason(league_id=11, season_year=2026)]


def test_list_daily_matches_handles_errors_and_missing_fixture(monkeypatch) -> None:
    leagues = [
        LeagueSeason(league_id=1, season_year=2024),
        LeagueSeason(league_id=2, season_year=2024),
    ]

    class _FixturesSometimesBad:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def get_daily_fixtures(
            self,
            for_date: str,
            *,
            league_id: int | None = None,
            season: int | None = None,
            tz_name: str = "Europe/Budapest",
        ) -> List[Mapping[str, Any]]:
            self.calls.append(int(league_id or 0))
            if league_id == 1:
                raise RuntimeError("fail first league")
            return [
                {"note": "missing ids"},  # should be skipped
                {
                    "fixture": {"id": "22", "status": {"short": "NS"}},
                    "teams": {"home": {"id": "9"}, "away": {"id": "10"}},
                },
            ]

    fixtures_stub = _FixturesSometimesBad()
    pipe = SelectionPipeline(_LeaguesFake([]), fixtures_stub, _OddsFake([]))

    rows = pipe.list_daily_matches("2025-01-01", leagues)
    assert fixtures_stub.calls == [1, 2]
    assert rows == [
        FixtureShort(
            fixture_id=22,
            league_id=2,
            season_year=2024,
            home_id=9,
            away_id=10,
            status="NS",
        )
    ]
