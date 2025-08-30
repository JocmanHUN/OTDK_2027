from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

from src.application.services.selection_pipeline import LeagueSeason, SelectionPipeline


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
