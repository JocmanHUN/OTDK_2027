from __future__ import annotations

from typing import Any, Dict, List, Mapping

from src.application.services.history_service import HistoryService, TeamAverages


class _FakeClient:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.calls: List[Mapping[str, Any]] = []

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        self.calls.append({"path": path, "params": params or {}})
        out = self.data.get(path, {})
        assert isinstance(out, dict)
        return out


def test_team_averages_parsing() -> None:
    payload = {
        "teams/statistics": {
            "response": {
                "fixtures": {"played": {"home": 15, "away": 14}},
                "goals": {
                    "for": {"average": {"home": "1.8", "away": "1.2"}},
                    "against": {"average": {"home": "1.0", "away": "1.4"}},
                },
            }
        }
    }
    svc = HistoryService(client=_FakeClient(payload))
    stats = svc.get_team_averages(1, 39, 2025)
    assert isinstance(stats, TeamAverages)
    assert stats.matches_home == 15 and stats.matches_away == 14
    assert stats.goals_for_home_avg == 1.8 and stats.goals_for_away_avg == 1.2
    assert stats.goals_against_home_avg == 1.0 and stats.goals_against_away_avg == 1.4


def test_simple_poisson_means() -> None:
    svc = HistoryService(client=_FakeClient({}))
    home = TeamAverages(10, 10, 2.0, 1.0, 1.0, 1.5)
    away = TeamAverages(10, 10, 1.1, 1.4, 1.3, 1.2)
    mu_h, mu_a = svc.simple_poisson_means(home, away)
    # mu_home = mean(home.GF_home=2.0, away.GA_away=1.2) = 1.6
    # mu_away = mean(away.GF_away=1.4, home.GA_home=1.0) = 1.2
    assert abs(mu_h - 1.6) < 1e-9 and abs(mu_a - 1.2) < 1e-9


def test_head_to_head_mapping() -> None:
    payload = {
        "fixtures/headtohead": {
            "response": [
                {
                    "fixture": {"date": "2025-02-01T18:00:00+02:00"},
                    "teams": {
                        "home": {"id": 1, "winner": True},
                        "away": {"id": 2, "winner": False},
                    },
                    "goals": {"home": 2, "away": 1},
                }
            ]
        }
    }
    svc = HistoryService(client=_FakeClient(payload))
    rows = svc.get_head_to_head(1, 2, last=5)
    assert len(rows) == 1 and rows[0]["result"] == "home"
    assert rows[0]["home_goals"] == 2 and rows[0]["away_goals"] == 1


def test_elo_input_defaults() -> None:
    svc = HistoryService(client=_FakeClient({}))
    elo = svc.elo_input(10, 20)
    assert elo.home_team_id == 10 and elo.away_team_id == 20
    assert elo.init_rating_home == 1500.0 and elo.home_advantage == 100.0
