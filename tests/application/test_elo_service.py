from __future__ import annotations

from typing import Any, cast

from src.application.services.elo_service import EloService


class _HistFake:
    def __init__(
        self,
        fixtures_by_key: dict[tuple[int, int], list[dict[str, Any]]],
        main_league: dict[tuple[int, int], int | None],
    ) -> None:
        self._fixtures = fixtures_by_key
        self._main = main_league

    def get_league_finished_fixtures(
        self, league_id: int, season: int
    ) -> list[dict[str, Any]]:  # noqa: D401
        return list(self._fixtures.get((int(league_id), int(season)), []))

    def get_team_main_league(self, team_id: int, season: int) -> int | None:  # noqa: D401
        return self._main.get((int(team_id), int(season)))


def test_elo_seeding_from_prev_and_baseline() -> None:
    # League 39 (PL) baseline 1600, H=80.
    league = 39
    prev_season = 2024
    curr_season = 2025
    home, away = 1, 2

    # Previous season: one FT home win 1-0
    fixtures_prev = [
        {
            "date_utc": None,
            "home_id": home,
            "away_id": away,
            "home_goals": 1,
            "away_goals": 0,
        }
    ]
    fixtures_curr: list[dict[str, Any]] = []

    hist = _HistFake(
        fixtures_by_key={
            (league, prev_season): fixtures_prev,
            (league, curr_season): fixtures_curr,
        },
        main_league={},
    )
    svc = EloService(history=cast(Any, hist))

    ratings_prev = svc.get_league_ratings(league, prev_season)
    # Home should gain some points over baseline 1600
    assert ratings_prev[home] > 1600.0
    assert ratings_prev[away] < 1600.0

    ratings_curr = svc.get_league_ratings(league, curr_season)
    # Seeds should be between prev rating and baseline due to regression (w=0.7)
    assert ratings_prev[home] > ratings_curr[home] > 1600.0
    assert ratings_prev[away] < ratings_curr[away] < 1600.0


def test_elo_promotion_offset_for_new_team() -> None:
    league = 39  # tier 1
    season_prev = 2024
    season_curr = 2025
    team_new = 3
    # No prev fixtures in target league; current season has at least one fixture containing the team,
    # so EloService considers it part of the league.
    fixtures_prev: list[dict[str, Any]] = []
    fixtures_curr = [
        {
            "date_utc": None,
            "home_id": team_new,
            "away_id": 99,
            "home_goals": 0,
            "away_goals": 0,
        }
    ]
    # Team's main league in previous season: 40 (ENG Championship) which is tier 2 -> promotion
    hist = _HistFake(
        fixtures_by_key={
            (league, season_prev): fixtures_prev,
            (league, season_curr): fixtures_curr,
        },
        main_league={(team_new, season_prev): 40},
    )
    svc = EloService(history=cast(Any, hist))
    ratings = svc.get_league_ratings(league, season_curr)
    # Promotion offset seeds to 1600 - 30 = 1570, then one FT draw updates rating slightly downward
    # because expected home > 0.5 due to home advantage.
    seed = 1600.0 - 30.0
    delta = seed - 1600.0 + 80.0  # H from tier config for PL
    expected = 1.0 / (1.0 + 10.0 ** (-delta / 400.0))
    new_rating = seed + 20.0 * (0.5 - expected)
    assert abs(ratings[team_new] - new_rating) < 1e-6


def test_elo_mov_scaling_increases_change() -> None:
    league = 39
    season = 2024
    home, away = 7, 8

    # Single match scenarios: 1-0 vs 3-0; same initial ratings
    fx_1_0 = [
        {"date_utc": None, "home_id": home, "away_id": away, "home_goals": 1, "away_goals": 0}
    ]
    fx_3_0 = [
        {"date_utc": None, "home_id": home, "away_id": away, "home_goals": 3, "away_goals": 0}
    ]

    hist_a = _HistFake(fixtures_by_key={(league, season): fx_1_0}, main_league={})
    hist_b = _HistFake(fixtures_by_key={(league, season): fx_3_0}, main_league={})

    elo_a = EloService(history=cast(Any, hist_a))
    elo_b = EloService(history=cast(Any, hist_b))

    r_a = elo_a.get_league_ratings(league, season)
    r_b = elo_b.get_league_ratings(league, season)

    gain_a = r_a[home] - 1600.0
    gain_b = r_b[home] - 1600.0
    assert gain_b > gain_a  # bigger GD → bigger rating gain
