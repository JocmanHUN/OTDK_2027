from __future__ import annotations

from typing import Any

from src.application.services.features_service import FeaturesService


class _HistFake:
    def __init__(self, home_rows: list[dict[str, Any]], away_rows: list[dict[str, Any]]) -> None:
        self._home = home_rows
        self._away = away_rows

    def get_recent_team_stats(
        self, team_id: int, league_id: int, season: int, last: int, *, only_finished: bool = True
    ) -> list[dict[str, Any]]:
        return self._home if team_id == 1 else self._away


def test_features_diff_and_aggregates() -> None:
    home_rows = [
        {
            "goals_for": 2,
            "goals_against": 1,
            "stats": {"ball possession": 55.0, "shots on target": 6.0},
        },
        {
            "goals_for": 1,
            "goals_against": 0,
            "stats": {"ball possession": 53.0, "shots on target": 5.0},
        },
    ]
    away_rows = [
        {
            "goals_for": 0,
            "goals_against": 1,
            "stats": {"ball possession": 45.0, "shots on target": 3.0},
        },
        {
            "goals_for": 1,
            "goals_against": 1,
            "stats": {"ball possession": 47.0, "shots on target": 4.0},
        },
    ]

    svc = FeaturesService(_HistFake(home_rows, away_rows))
    feats = svc.build_features(home_team_id=1, away_team_id=2, league_id=39, season=2025, last=2)

    # Aggregates
    # Home GF avg = (2+1)/2 = 1.5; Away GF avg = (0+1)/2 = 0.5 → diff=1.0
    assert abs(feats["diff_goals_for_avg"] - 1.0) < 1e-9
    # GA avg: Home (1+0)/2=0.5; Away (1+1)/2=1.0 → diff=-0.5
    assert abs(feats["diff_goals_against_avg"] - (-0.5)) < 1e-9
    # PPG: Home (3+3)/2=3.0; Away (0+1)/2=0.5 → diff=2.5
    assert abs(feats["diff_points_per_game"] - 2.5) < 1e-9

    # Stats
    # Possession: Home avg 54, Away avg 46 → diff=8
    assert abs(feats["diff_ball possession"] - 8.0) < 1e-9
    # Shots on target: Home avg 5.5, Away 3.5 → diff=2
    assert abs(feats["diff_shots on target"] - 2.0) < 1e-9


def test_features_skips_non_numeric_stats() -> None:
    svc = FeaturesService(
        _HistFake(
            [{"goals_for": 1, "goals_against": 0, "stats": {"bad": "x"}}],
            [{"goals_for": 0, "goals_against": 1, "stats": {"good": 2}}],
        )
    )
    feats = svc.build_features(home_team_id=1, away_team_id=2, league_id=1, season=2024, last=1)
    # Non-numeric "bad" should be skipped, only "good" appears as diff_good
    assert "diff_bad" not in feats
    assert "diff_good" in feats
