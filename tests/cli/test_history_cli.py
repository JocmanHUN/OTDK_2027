# mypy: ignore-errors

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


def test_fmt_helpers_and_print(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import src.cli.history as history_cli

    dt = datetime(2024, 1, 2, 3, 4, tzinfo=timezone.utc)
    assert "2024-01-02T04:04" in history_cli._fmt_dt(dt, "Europe/Budapest")
    # Invalid timezone falls back to isoformat
    assert history_cli._fmt_dt(dt, "Bad/Zone").startswith("2024-01-02T")
    assert history_cli._fmt_dt("raw", "UTC") == "raw"

    history_cli._print_h2h([], "UTC")
    assert "No head-to-head" in capsys.readouterr().out

    history_cli._print_h2h(
        [
            {
                "date_utc": dt,
                "home_id": 1,
                "away_id": 2,
                "home_goals": 0,
                "away_goals": 1,
                "result": "2",
            }
        ],
        "UTC",
    )
    assert "1 vs 2" in capsys.readouterr().out


def test_format_all_stats_formats_numbers() -> None:
    import src.cli.history as history_cli

    lines = history_cli._format_all_stats(
        {
            "ball possession": 55,
            "passes %": 80.0,
            "shots_on_target": 3.7,
            "note": "raw",
        }
    )
    joined = " | ".join(lines)
    assert "ball possession: 55%" in joined
    assert "passes %: 80%" in joined
    assert "shots on target: 3.70" in joined
    assert "note: raw" in joined


def test_history_cli_recent_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import src.cli.history as history_cli

    rows = [
        {
            "date_utc": datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc),
            "home_away": "H",
            "opponent_id": 9,
            "goals_for": 1,
            "goals_against": 0,
        }
    ]

    class _Svc:
        def get_head_to_head(self, *args, **kwargs):  # pragma: no cover
            return []

        def get_team_averages(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

        def simple_poisson_means(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

        def get_recent_team_stats(self, *args, **kwargs):
            return rows

    monkeypatch.setattr(history_cli, "HistoryService", _Svc)

    rc = history_cli.main(
        ["recent", "--team", "1", "--league", "2", "--season", "2024", "--last", "1", "--json"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert '"date_utc": "2024-05-01T12:00:00+00:00"' in out


def test_history_cli_recent_all_stats(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import src.cli.history as history_cli

    rows = [
        {
            "date_utc": datetime(2024, 6, 1, 18, 30, tzinfo=timezone.utc),
            "home_away": "A",
            "opponent_id": 77,
            "goals_for": 2,
            "goals_against": 1,
            "stats": {"ball possession": 60, "shots": 10},
            "all_stats": {1: {"xg": 1.5}, 2: {"xg": 0.7}},
        }
    ]

    class _Svc:
        def get_head_to_head(self, *args, **kwargs):  # pragma: no cover
            return []

        def get_team_averages(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

        def simple_poisson_means(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

        def get_recent_team_stats(self, *args, **kwargs):
            return rows

    monkeypatch.setattr(history_cli, "HistoryService", _Svc)

    rc = history_cli.main(
        [
            "recent",
            "--team",
            "1",
            "--league",
            "2",
            "--season",
            "2024",
            "--last",
            "1",
            "--timezone",
            "UTC",
            "--all-stats",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "vs 77" in out and "team=1 stats" in out


def test_history_cli_recent_no_rows(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import src.cli.history as history_cli

    class _Svc:
        def get_head_to_head(self, *args, **kwargs):  # pragma: no cover
            return []

        def get_team_averages(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

        def simple_poisson_means(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

        def get_recent_team_stats(self, *args, **kwargs):
            return []

    monkeypatch.setattr(history_cli, "HistoryService", _Svc)

    rc = history_cli.main(
        ["recent", "--team", "1", "--league", "2", "--season", "2024", "--last", "1"]
    )
    assert rc == 0
    assert "No fixtures found." in capsys.readouterr().out
