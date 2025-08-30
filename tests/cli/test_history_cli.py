from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping

import pytest


class _FakeHistoryService:
    def __init__(self) -> None:
        self._h2h: list[dict[str, Any]] = []
        self._stats: Dict[int, dict[str, Any]] = {}

    # H2H
    def get_head_to_head(
        self, home_team_id: int, away_team_id: int, last: int = 20
    ) -> List[dict[str, Any]]:
        return self._h2h

    # Stats
    class _TA:
        def __init__(self, d: Mapping[str, Any]) -> None:
            self.matches_home = d["matches_home"]
            self.matches_away = d["matches_away"]
            self.goals_for_home_avg = d["goals_for_home_avg"]
            self.goals_for_away_avg = d["goals_for_away_avg"]
            self.goals_against_home_avg = d["goals_against_home_avg"]
            self.goals_against_away_avg = d["goals_against_away_avg"]

    def get_team_averages(self, team_id: int, league_id: int, season: int):  # type: ignore[no-untyped-def]
        return self._TA(self._stats[team_id])

    def simple_poisson_means(self, home, away) -> tuple[float, float]:  # type: ignore[no-untyped-def]
        mu_home = (home.goals_for_home_avg + away.goals_against_away_avg) / 2.0
        mu_away = (away.goals_for_away_avg + home.goals_against_home_avg) / 2.0
        return (mu_home, mu_away)


def test_history_cli_h2h(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import src.cli.history as history_cli

    fake = _FakeHistoryService()
    fake._h2h = [
        {
            "date_utc": datetime(2025, 2, 1, 16, 0, tzinfo=timezone.utc),
            "home_id": 1,
            "away_id": 2,
            "home_goals": 2,
            "away_goals": 1,
            "result": "home",
        }
    ]
    monkeypatch.setattr(history_cli, "HistoryService", lambda: fake)

    rc = history_cli.main(["h2h", "1", "2", "--last", "5", "--timezone", "UTC"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 vs 2" in out and "2-1" in out and "home" in out


def test_history_cli_stats_poisson(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import src.cli.history as history_cli

    fake = _FakeHistoryService()
    fake._stats = {
        10: {
            "matches_home": 10,
            "matches_away": 10,
            "goals_for_home_avg": 2.0,
            "goals_for_away_avg": 1.2,
            "goals_against_home_avg": 1.0,
            "goals_against_away_avg": 1.1,
        },
        20: {
            "matches_home": 10,
            "matches_away": 10,
            "goals_for_home_avg": 1.3,
            "goals_for_away_avg": 1.4,
            "goals_against_home_avg": 1.0,
            "goals_against_away_avg": 1.2,
        },
    }
    monkeypatch.setattr(history_cli, "HistoryService", lambda: fake)

    rc = history_cli.main(
        ["stats", "--team", "10", "--league", "39", "--season", "2025", "--opponent", "20"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Team averages:" in out
    assert "Poisson means vs opponent" in out
