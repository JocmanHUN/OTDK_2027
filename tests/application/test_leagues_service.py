from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping

import pytest

from src.application.services.leagues_service import LeaguesService


class _FakeClient:
    def __init__(self, responses: Dict[str, Dict[str, Any]]):
        self.responses = responses
        self.calls: List[Dict[str, Any]] = []

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> object:
        if params and params.get("current"):
            key = f"{path}?current"
        else:
            assert params is not None and "season" in params
            key = f"{path}?season={params.get('season')}"
        self.calls.append({"path": path, "params": params})
        return self.responses[key]


def _payload(leagues: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"response": leagues}


def test_get_current_leagues_filters_coverage() -> None:
    # Two leagues, only second has odds+stats on current season
    responses = {
        "leagues?current": _payload(
            [
                {
                    "league": {"id": 1, "name": "L1"},
                    "country": {"name": "X"},
                    "seasons": [
                        {
                            "year": 2024,
                            "current": True,
                            "coverage": {"odds": False, "fixtures": {"statistics": True}},
                        }
                    ],
                },
                {
                    "league": {"id": 2, "name": "L2"},
                    "country": {"name": "Y"},
                    "seasons": [
                        {
                            "year": 2024,
                            "current": True,
                            "coverage": {"odds": True, "fixtures": {"statistics": True}},
                        }
                    ],
                },
            ]
        )
    }
    client = _FakeClient(responses)
    svc = LeaguesService(client=client, ttl_seconds=60)

    result = svc.get_current_leagues()
    assert len(result) == 1
    assert result[0]["league_id"] == 2
    assert result[0]["has_odds"] and result[0]["has_stats"]


def test_get_leagues_for_season_filters_and_caches() -> None:
    responses = {
        "leagues?season=2023": _payload(
            [
                {
                    "league": {"id": 10, "name": "LA"},
                    "country": {"name": "A"},
                    "seasons": [
                        {
                            "year": 2022,
                            "coverage": {"odds": True, "fixtures": {"statistics": True}},
                        },
                        {
                            "year": 2023,
                            "coverage": {"odds": True, "fixtures": {"statistics": False}},
                        },
                    ],
                },
                {
                    "league": {"id": 11, "name": "LB"},
                    "country": {"name": "B"},
                    "seasons": [
                        {
                            "year": 2023,
                            "coverage": {"odds": True, "fixtures": {"statistics": True}},
                        },
                    ],
                },
            ]
        )
    }
    client = _FakeClient(responses)
    svc = LeaguesService(client=client, ttl_seconds=60)

    result = svc.get_leagues_for_season(2023)
    assert len(result) == 1
    assert result[0]["league_id"] == 11
    assert result[0]["season_year"] == 2023

    # Cached: second call should not trigger another client.get
    _ = svc.get_leagues_for_season(2023)
    assert len(client.calls) == 1


def test_cache_ttl_expiry_triggers_refetch(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "leagues?current": _payload(
            [
                {
                    "league": {"id": 1, "name": "L1"},
                    "country": {"name": "X"},
                    "seasons": [
                        {
                            "year": 2024,
                            "current": True,
                            "coverage": {"odds": True, "fixtures": {"statistics": True}},
                        }
                    ],
                }
            ]
        )
    }

    client = _FakeClient(responses)
    svc = LeaguesService(client=client, ttl_seconds=0.01)

    # First call fills cache
    _ = svc.get_current_leagues()
    # Fast second call uses cache
    _ = svc.get_current_leagues()
    assert len(client.calls) == 1

    # Simulate time passing beyond TTL
    real_monotonic = time.monotonic

    def fake_monotonic() -> float:
        return real_monotonic() + 1.0

    monkeypatch.setattr("time.monotonic", fake_monotonic)
    _ = svc.get_current_leagues()
    assert len(client.calls) == 2
