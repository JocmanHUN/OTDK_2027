# mypy: ignore-errors

from __future__ import annotations

from typing import Any, Mapping

import pytest

from src.application.services.history_service import (
    EloInput,
    HistoryService,
    TeamAverages,
    _extract_xg,
    _normalize_stat_value,
    _safe_float,
    _safe_int,
    _to_utc,
)
from src.infrastructure.api_football_client import APIError


class DummyClient:
    def __init__(self, payloads: dict[str, Any]) -> None:
        self.payloads = payloads
        self.calls: dict[str, int] = {}

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        self.calls[path] = self.calls.get(path, 0) + 1
        key = path
        if path == "fixtures" and params and "id" in params:
            key = f"{path}:{params.get('id')}"
        return self.payloads.get(key)


def test_helpers_basic() -> None:
    assert _safe_float("1.5") == 1.5
    assert _safe_float(None) == 0.0
    assert _safe_int("3") == 3
    assert _safe_int("bad") is None
    assert _normalize_stat_value("55%") == 55.0
    assert _normalize_stat_value("x") is None
    assert _normalize_stat_value(1.2) == 1.2
    assert _extract_xg({"xg_for": 1.0}) == 1.0
    assert _extract_xg({"expected goals": "2.1"}) == 2.1
    assert _extract_xg({"shots": 3}) is None
    dt = _to_utc("2024-01-01T12:00:00Z")
    assert dt.tzinfo is not None and dt.utcoffset().total_seconds() == 0


def test_get_head_to_head_filters_friendlies() -> None:
    payload = {
        "response": [
            {
                "league": {"type": "Friendly", "name": "Friendly Cup"},
                "fixture": {"date": "2024-01-01T12:00:00Z"},
                "teams": {"home": {"id": 1, "winner": True}, "away": {"id": 2, "winner": False}},
                "goals": {"home": 2, "away": 0},
            },
            {
                "league": {"type": "League", "name": "Premier"},
                "fixture": {"date": "2024-01-02T12:00:00Z"},
                "teams": {"home": {"id": 1, "winner": False}, "away": {"id": 2, "winner": False}},
                "goals": {"home": 1, "away": 1},
            },
        ]
    }
    svc = HistoryService(client=DummyClient({"fixtures/headtohead": payload}))
    rows = svc.get_head_to_head(1, 2)
    assert len(rows) == 1
    assert rows[0]["result"] == "draw"


def test_get_team_averages_cached() -> None:
    payload = {
        "response": {
            "fixtures": {"played": {"home": 5, "away": 7}},
            "goals": {
                "for": {"average": {"home": "1.6", "away": "1.2"}},
                "against": {"average": {"home": "1.1", "away": "1.3"}},
            },
        }
    }
    client = DummyClient({"teams/statistics": payload})
    svc = HistoryService(client=client)
    avg1 = svc.get_team_averages(1, 2, 2024)
    avg2 = svc.get_team_averages(1, 2, 2024)  # cached
    assert client.calls["teams/statistics"] == 1
    assert avg1 == avg2
    assert isinstance(avg1, TeamAverages)


def test_simple_poisson_means_and_elo_input() -> None:
    svc = HistoryService(client=DummyClient({}))
    home = TeamAverages(1, 1, 1.4, 1.0, 1.1, 1.2)
    away = TeamAverages(1, 1, 1.0, 0.8, 1.3, 1.1)
    mu_h, mu_a = svc.simple_poisson_means(home, away)
    assert mu_h > mu_a
    elo_in = svc.elo_input(1, 2, init_rating=1500, home_advantage=80)
    assert isinstance(elo_in, EloInput) and elo_in.home_advantage == 80


def test_get_recent_team_stats_excludes_friendlies_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Season 2024 empty, 2023 has one friendly and one league match
    fixtures_payload_2024 = {"response": []}
    fixtures_payload_2023 = {
        "response": [
            {
                "fixture": {"id": 10, "timestamp": 1700000000, "status": {"short": "FT"}},
                "league": {"id": 99, "type": "friendly", "name": "Friendly"},
                "teams": {"home": {"id": 1}, "away": {"id": 2}},
                "goals": {"home": 1, "away": 0},
            },
            {
                "fixture": {"id": 11, "timestamp": 1690000000, "status": {"short": "FT"}},
                "league": {"id": 99, "type": "league", "name": "League"},
                "teams": {"home": {"id": 2}, "away": {"id": 1}},
                "goals": {"home": 0, "away": 2},
            },
        ]
    }

    client = DummyClient({"fixtures": fixtures_payload_2024})

    def _client_get(path: str, params: Mapping[str, Any] | None = None) -> Any:
        # simulate two seasons
        if path == "fixtures":
            season = params.get("season") if params else None
            if str(season) == "2024":
                return fixtures_payload_2024
            if str(season) == "2023":
                return fixtures_payload_2023
            return {"response": []}
        return {}

    client.get = _client_get  # type: ignore[assignment]

    svc = HistoryService(client=client)
    monkeypatch.setattr(
        svc, "get_fixture_statistics", lambda fixture_id: {1: {"xg": 1.1}, 2: {"xg": 0.8}}
    )

    rows1 = svc.get_recent_team_stats(1, 99, 2024, 2)
    rows2 = svc.get_recent_team_stats(1, 99, 2024, 2)
    assert len(rows1) == 1  # friendly excluded
    assert rows1[0]["fixture_id"] == 11
    assert rows1[0]["xg_against"] == 0.8
    assert rows2 == rows1  # cached


def test_get_recent_team_scores_basic() -> None:
    fixtures_payload = {
        "response": [
            {
                "fixture": {"id": 20, "timestamp": 1700000000, "status": {"short": "FT"}},
                "league": {"id": 1, "type": "league"},
                "teams": {"home": {"id": 5}, "away": {"id": 6}},
                "goals": {"home": 2, "away": 1},
            }
        ]
    }
    client = DummyClient({"fixtures": fixtures_payload})
    svc = HistoryService(client=client)
    rows = svc.get_recent_team_scores(5, 1, 2024, 1)
    assert len(rows) == 1 and rows[0]["goals_for"] == 2 and rows[0]["home_away"] == "H"


def test_league_goal_means_prior_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DummyClient({})
    svc = HistoryService(client=client)

    called = {"count": 0}

    def _fake_league(league_id: int, season: int) -> list[dict[str, Any]]:
        called["count"] += 1
        return []

    monkeypatch.setattr(svc, "get_league_finished_fixtures", _fake_league)
    prior = svc.league_goal_means(1, 2024)
    assert prior == (1.35, 1.15)
    _ = svc.league_goal_means(1, 2024)  # cached
    assert called["count"] == 1


def test_get_fixture_statistics_parses_and_caches() -> None:
    payload = {
        "response": [
            {
                "team": {"id": 1},
                "statistics": [
                    {"type": "xG", "value": 1.1},
                    {"type": "Ball Possession", "value": "55%"},
                ],
            },
            {
                "team": {"id": 2},
                "statistics": [
                    {"type": "Expected Goals", "value": 0.9},
                    {"type": "Shots on Target", "value": 3},
                ],
            },
        ]
    }
    client = DummyClient({"fixtures/statistics": payload})
    svc = HistoryService(client=client)
    stats1 = svc.get_fixture_statistics(10)
    stats2 = svc.get_fixture_statistics(10)
    assert client.calls["fixtures/statistics"] == 1
    assert stats1[1]["ball possession"] == 55.0
    assert stats1[2]["expected goals"] == 0.9
    assert stats2 == stats1


def test_get_fixture_result_label_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    payload_win = {
        "response": [
            {
                "fixture": {"status": {"short": "FT"}},
                "goals": {"home": 2, "away": 1},
            }
        ]
    }
    payload_score_only = {
        "response": [
            {
                "fixture": {"status": {"short": "FT"}},
                "goals": {"home": None, "away": None},
                "score": {"fulltime": {"home": 0, "away": 0}},
            }
        ]
    }
    client = DummyClient({"fixtures:1": payload_win, "fixtures:2": payload_score_only})
    svc = HistoryService(client=client)
    assert svc.get_fixture_result_label(1) == "1"
    assert svc.get_fixture_result_label(2) == "X"
    assert svc.get_fixture_result_label(3) is None  # missing response

    class BoomClient:
        def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
            raise APIError("rate", status_code=429)

    with pytest.raises(APIError):
        HistoryService(client=BoomClient()).get_fixture_result_label(1)

    class ErrClient:
        def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
            raise RuntimeError("boom")

    assert HistoryService(client=ErrClient()).get_fixture_result_label(1) is None


def test_get_team_main_league() -> None:
    payload = {
        "response": [
            {"league": {"id": 1}},
            {"league": {"id": 1}},
            {"league": {"id": 2}},
        ]
    }
    client = DummyClient({"fixtures": payload})
    svc = HistoryService(client=client)
    assert svc.get_team_main_league(1, 2024) == 1


# ---- merged additional coverage tests ----


class CacheStub:
    def __init__(self, value=None) -> None:
        self.value = value
        self.set_calls: list[tuple[Any, Any]] = []

    def get(self, key):
        return self.value

    def set(self, key, value):
        self.set_calls.append((key, value))
        self.value = value


class QueueClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, Mapping[str, Any] | None]] = []

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        self.calls.append((path, params or {}))
        if self.responses:
            return self.responses.pop(0)
        return {}


def test_head_to_head_filters_friendlies_and_parses_extra() -> None:
    payload = {
        "response": [
            {  # friendly should be skipped
                "league": {"type": "friendly", "name": "Friendly Cup"},
                "fixture": {"date": "2024-01-01T10:00:00+00:00"},
                "teams": {"home": {"id": 1}, "away": {"id": 2}},
                "goals": {"home": 1, "away": 0},
            },
            {
                "league": {"type": "league", "name": "League"},
                "fixture": {"date": "2024-01-02T10:00:00+00:00"},
                "teams": {"home": {"id": 1, "winner": True}, "away": {"id": 2}},
                "goals": {"home": 2, "away": 1},
            },
        ]
    }
    svc = HistoryService(client=QueueClient([payload]))
    rows = svc.get_head_to_head(1, 2, last=2)
    assert len(rows) == 1
    assert rows[0]["result"] == "home"


def test_team_averages_cache_and_parse_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "response": {
            "fixtures": {"played": {"home": "10", "away": "8"}},
            "goals": {
                "for": {"average": {"home": "1.5", "away": "1.2"}},
                "against": {"average": {"home": "1.1", "away": "0.9"}},
            },
        }
    }
    svc = HistoryService(client=QueueClient([payload]))
    avg = svc.get_team_averages(1, 1, 2024)
    assert avg.matches_home == 10 and avg.goals_for_home_avg == 1.5
    svc._cache_team_avg = CacheStub(avg)  # type: ignore[assignment]
    cached = svc.get_team_averages(1, 1, 2024)
    assert cached == avg


def test_recent_team_stats_cache_and_friendlies_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = HistoryService(client=QueueClient([]))
    svc._cache_recent_stats = CacheStub([{"cached": True}])  # type: ignore[assignment]
    out = svc.get_recent_team_stats(1, 1, 2024, 2)
    assert out == [{"cached": True}]

    def fake_fetch(team_id, league_id, season, only_finished=True):
        return [
            {
                "id": 10,
                "timestamp": 1700000000,
                "league": {"type": "league", "name": "L"},
                "teams": {"home": {"id": 1}, "away": {"id": 2}},
                "goals": {"home": 1, "away": 0},
            },
            {
                "id": 11,
                "timestamp": 1700000100,
                "league": {"type": "friendly", "name": "Friendly"},
                "teams": {"home": {"id": 1}, "away": {"id": 3}},
                "goals": {"home": 0, "away": 0},
            },
        ]

    svc._cache_recent_stats = CacheStub(None)  # type: ignore[assignment]
    svc._cache_fixture_stats = CacheStub({})  # no stats
    monkeypatch.setattr(svc, "_fetch_team_fixtures_for_season", fake_fetch)
    rows = svc.get_recent_team_stats(1, 1, 2024, 1)
    assert len(rows) == 1 and rows[0]["fixture_id"] == 10


def test_recent_team_scores_empty_streak_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = HistoryService(client=QueueClient([]))

    def fake_fetch(team_id, league_id, season, only_finished=True):
        if season == 2024:
            return []
        return [
            {
                "id": 20,
                "timestamp": None,
                "date": "2024-01-01T00:00:00+00:00",
                "teams": {"home": {"id": 2}, "away": {"id": 1}},
                "goals": {"home": 0, "away": 1},
            }
        ]

    monkeypatch.setattr(svc, "_fetch_team_fixtures_for_season", fake_fetch)
    rows = svc.get_recent_team_scores(1, 1, 2024, 1)
    assert rows and rows[0]["home_away"] == "A"


def test_league_goal_means_prior_and_cache_extra() -> None:
    svc = HistoryService(client=QueueClient([{}]))
    svc._cache_league_means = CacheStub(None)  # type: ignore[assignment]
    means = svc.league_goal_means(1, 2024)
    assert means == (1.35, 1.15)
    svc._cache_league_means = CacheStub((0.9, 0.8))  # type: ignore[assignment]
    assert svc.league_goal_means(1, 2024) == (0.9, 0.8)


def test_get_league_finished_fixtures_paging_extra() -> None:
    first = {
        "response": [
            {
                "fixture": {"id": 1, "date": "2024-01-01T00:00:00+00:00"},
                "teams": {"home": {"id": 1}, "away": {"id": 2}},
                "goals": {"home": 1, "away": 0},
            }
        ],
        "paging": {"total": 2},
    }
    second = {
        "response": [
            {
                "fixture": {"id": 2, "date": "2024-01-02T00:00:00+00:00"},
                "teams": {"home": {"id": 2}, "away": {"id": 1}},
                "goals": {"home": 0, "away": 1},
            }
        ]
    }
    svc = HistoryService(client=QueueClient([first, second]))
    out = svc.get_league_finished_fixtures(1, 2024)
    ids = [r["fixture_id"] for r in out]
    assert ids == [1, 2]


def test_fixture_statistics_missing_xg_prints_extra(capsys: pytest.CaptureFixture[str]) -> None:
    payload = {
        "response": [
            {"team": {"id": 1}, "statistics": [{"type": "Shots on Target", "value": "5"}]},
            {"team": {"id": 2}, "statistics": [{"type": "Ball Possession", "value": "55%"}]},
        ]
    }
    svc = HistoryService(client=QueueClient([payload]))
    stats = svc.get_fixture_statistics(10)
    out = capsys.readouterr().out
    assert "[XG-MISS]" in out
    assert stats[2]["ball possession"] == 55.0


def test_fixture_statistics_cache_hit_extra() -> None:
    svc = HistoryService(client=QueueClient([]))
    svc._cache_fixture_stats = CacheStub({1: {"a": 1}})  # type: ignore[assignment]
    assert svc.get_fixture_statistics(5) == {1: {"a": 1}}


def test_fixture_result_label_paths_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = HistoryService(client=QueueClient([]))
    svc._client = QueueClient([123])  # type: ignore[assignment]
    assert svc.get_fixture_result_label(1) is None

    payload = {
        "response": [
            {
                "fixture": {"status": {"short": "FT"}},
                "goals": {"home": None, "away": None},
                "score": {"fulltime": {"home": 2, "away": 3}},
            }
        ]
    }
    svc._client = QueueClient([payload])  # type: ignore[assignment]
    assert svc.get_fixture_result_label(2) == "2"

    class ErrClient:
        def get(self, path, params=None):
            raise APIError("rate", status_code=429)

    svc._client = ErrClient()  # type: ignore[assignment]
    with pytest.raises(APIError):
        svc.get_fixture_result_label(3)


def test_team_main_league_counts_extra() -> None:
    payload = {
        "response": [
            {"league": {"id": 1}},
            {"league": {"id": 2}},
            {"league": {"id": 2}},
        ]
    }
    svc = HistoryService(client=QueueClient([payload]))
    assert svc.get_team_main_league(1, 2024) == 2


def test_helpers_normalize_and_xg_extract_extra() -> None:
    assert _safe_float("bad") == 0.0
    assert _safe_int("x") is None
    assert _normalize_stat_value("55%") == 55.0
    assert _normalize_stat_value("bad%") is None
    assert _normalize_stat_value(None) is None
    assert _extract_xg({"expected goals": "1.2"}) == 1.2
    assert _extract_xg({"shots": 5}) is None
    dt = _to_utc("2024-01-01T00:00:00Z")
    assert dt.tzinfo is not None and dt.utcoffset().total_seconds() == 0


# ---- original basic tests merged ----
class _FakeClient:
    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.calls: list[Mapping[str, Any]] = []

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append({"path": path, "params": params or {}})
        out = self.data.get(path, {})
        assert isinstance(out, dict)
        return out


def test_team_averages_parsing_original() -> None:
    payload = {
        "teams/statistics": {
            "response": {
                "fixtures": {"played": {"home": 15, "away": 14}},
                "goals": {
                    "for": {"average": {"home": 2.1, "away": 1.3}},
                    "against": {"average": {"home": 1.0, "away": 0.9}},
                },
            }
        }
    }
    svc = HistoryService(client=_FakeClient(payload))
    averages = svc.get_team_averages(1, 1, 2024)
    assert averages.matches_home == 15
    assert averages.goals_for_home_avg == 2.1


def test_simple_poisson_means_original() -> None:
    h = TeamAverages(10, 8, 1.2, 1.0, 0.8, 0.7)
    a = TeamAverages(12, 10, 0.9, 1.1, 1.0, 0.9)
    mu_home, mu_away = HistoryService(client=_FakeClient({})).simple_poisson_means(h, a)
    assert mu_home > 0 and mu_away > 0


# ---- extra coverage gaps ----
def test_head_to_head_away_winner() -> None:
    payload = {
        "fixtures/headtohead": {
            "response": [
                {
                    "league": {"type": "league", "name": "X"},
                    "fixture": {"date": "2024-01-01T00:00:00Z"},
                    "teams": {
                        "home": {"id": 1, "winner": False},
                        "away": {"id": 2, "winner": True},
                    },
                    "goals": {"home": 0, "away": 1},
                }
            ]
        }
    }
    svc = HistoryService(client=_FakeClient(payload))
    rows = svc.get_head_to_head(1, 2, last=1, exclude_friendlies=False)
    assert rows[0]["result"] == "away"


def test_team_averages_handles_bad_payload_and_cache_error(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"teams/statistics": {"response": {"fixtures": None}}}
    svc = HistoryService(client=_FakeClient(payload))
    monkeypatch.setattr(
        svc._cache_team_avg,
        "set",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    avg = svc.get_team_averages(1, 1, 2024)
    assert avg.matches_home == 0 and avg.goals_for_home_avg == 0.0


class _RecentStatsStub(HistoryService):
    def __init__(self, fixtures_by_season: dict[int, list[dict[str, Any]]]):
        super().__init__(client=_FakeClient({}))
        self.fixtures_by_season = fixtures_by_season

    def _fetch_team_fixtures_for_season(
        self, team_id: int, league_id: int, season: int, *, only_finished: bool = True
    ) -> list[dict[str, Any]]:
        return list(self.fixtures_by_season.get(season, []))

    def get_fixture_statistics(self, fixture_id: int) -> dict[int, dict[str, float]]:
        return {}


def test_recent_team_stats_break_and_cache_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fixtures = {
        2024: [
            {
                "id": 1,
                "timestamp": 100,
                "league": {},
                "teams": {"home": {"id": 1}, "away": {"id": 2}},
                "goals": {"home": 1, "away": 0},
            },
            {
                "id": 2,
                "timestamp": 90,
                "league": {},
                "teams": {"home": {"id": 1}, "away": {"id": 3}},
                "goals": {"home": 0, "away": 0},
            },
        ]
    }
    svc = _RecentStatsStub(fixtures)
    monkeypatch.setattr(
        svc._cache_recent_stats,
        "set",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cache")),
    )
    rows = svc.get_recent_team_stats(
        team_id=1, league_id=1, season=2024, last=1, exclude_friendlies=False
    )
    assert len(rows) == 1 and rows[0]["fixture_id"] == 1


class _RecentScoresStub(HistoryService):
    def __init__(self, fixtures_by_season: dict[int, list[dict[str, Any]]]):
        super().__init__(client=_FakeClient({}))
        self.fixtures_by_season = fixtures_by_season
        self.call_count = 0

    def _fetch_team_fixtures_for_season(
        self, team_id: int, league_id: int, season: int, *, only_finished: bool = True
    ) -> list[dict[str, Any]]:
        self.call_count += 1
        return list(self.fixtures_by_season.get(season, []))


def test_recent_team_scores_stops_after_two_empty_seasons() -> None:
    svc = _RecentScoresStub({})
    rows = svc.get_recent_team_scores(team_id=1, league_id=1, season=2024, last=2)
    assert rows == []
    assert svc.call_count >= 2


def test_recent_team_scores_break_when_enough() -> None:
    fixtures = {
        2024: [
            {
                "id": 10,
                "timestamp": 50,
                "teams": {"home": {"id": 1}, "away": {"id": 2}},
                "goals": {"home": 1, "away": 0},
            },
            {
                "id": 11,
                "timestamp": 40,
                "teams": {"home": {"id": 1}, "away": {"id": 3}},
                "goals": {"home": 2, "away": 2},
            },
        ]
    }
    svc = _RecentScoresStub(fixtures)
    rows = svc.get_recent_team_scores(team_id=1, league_id=1, season=2024, last=1)
    assert len(rows) == 1 and rows[0]["fixture_id"] == 10


def test_league_goal_means_computation_and_cache() -> None:
    svc = HistoryService(client=_FakeClient({}))
    svc.get_league_finished_fixtures = lambda league_id, season: [
        {"home_goals": 1, "away_goals": 2},
        {"home_goals": 3, "away_goals": 1},
    ]
    mu_h, mu_a = svc.league_goal_means(1, 2024)
    assert mu_h == 2.0 and mu_a == 1.5


def test_get_league_finished_fixtures_handles_bad_paging() -> None:
    payload = {
        "fixtures": {
            "response": [
                {
                    "fixture": {"id": 1, "date": "2024-01-01T00:00:00Z"},
                    "teams": {"home": {"id": 1}, "away": {"id": 2}},
                    "goals": {"home": 1, "away": 0},
                }
            ],
            "paging": {"total": "bad"},
        }
    }
    svc = HistoryService(client=_FakeClient(payload))
    rows = svc.get_league_finished_fixtures(1, 2024)
    assert len(rows) == 1


def test_fetch_team_fixtures_filters_non_finished() -> None:
    payload = {
        "fixtures": {
            "response": [
                {
                    "fixture": {
                        "id": 1,
                        "status": {"short": "NS"},
                        "timestamp": 0,
                        "date": "2024-01-01T00:00:00Z",
                    },
                    "league": {},
                    "teams": {"home": {"id": 1}, "away": {"id": 2}},
                    "goals": {"home": 0, "away": 0},
                }
            ]
        }
    }
    svc = HistoryService(client=_FakeClient(payload))
    rows = svc._fetch_team_fixtures_for_season(
        team_id=1, league_id=1, season=2024, only_finished=True
    )
    assert rows == []


def test_fixture_statistics_handles_missing_team_and_cache_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "fixtures/statistics": {
            "response": [
                {"team": {"id": None}, "statistics": []},
                {"team": {"id": 2}, "statistics": [{"type": "xG", "value": "bad"}]},
            ]
        }
    }
    svc = HistoryService(client=_FakeClient(payload))
    monkeypatch.setattr(
        "src.application.services.history_service.json.dumps",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("no json")),
    )
    monkeypatch.setattr(
        svc._cache_fixture_stats,
        "set",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cache")),
    )
    stats = svc.get_fixture_statistics(99)
    assert 2 in stats and stats[2] == {}


def test_fixture_result_label_branches_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = HistoryService(client=_FakeClient({"fixtures": {"response": None}}))
    assert svc.get_fixture_result_label(1) is None

    svc2 = HistoryService(
        client=_FakeClient(
            {
                "fixtures": {
                    "response": [
                        {"fixture": {"status": {"short": "NS"}}, "goals": {"home": 1, "away": 0}},
                    ]
                }
            }
        )
    )
    assert svc2.get_fixture_result_label(1) is None

    bad_goals_payload = {
        "fixtures": {
            "response": [
                {
                    "fixture": {"status": {"short": "FT"}},
                    "goals": {"home": "bad", "away": "1"},
                    "score": {"penalty": object()},
                }
            ]
        }
    }
    svc3 = HistoryService(client=_FakeClient(bad_goals_payload))
    assert svc3.get_fixture_result_label(1) is None


def test_team_main_league_handles_missing_ids() -> None:
    payload = {
        "fixtures": {
            "response": [
                {"league": {"id": None}},
                {"league": {"id": None}},
            ]
        }
    }
    svc = HistoryService(client=_FakeClient(payload))
    assert svc.get_team_main_league(1, 2024) is None


def test_to_utc_and_extract_xg_edge_cases() -> None:
    dt = _to_utc("2024-01-01T00:00:00.000bad")
    assert dt.tzinfo is not None
    assert _extract_xg({"expectedGoals": "bad"}) is None
