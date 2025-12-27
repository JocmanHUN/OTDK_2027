# mypy: ignore-errors

from __future__ import annotations

import sys
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import src.application.services.odds_service as odds_service
from src.application.services.odds_service import OddsService, _extract_response_list, _safe_int


class DummyClient:
    def __init__(self, payload: Any):
        self.payload = payload
        self.calls = 0

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        self.calls += 1
        return self.payload


def test_extract_response_list_and_safe_int() -> None:
    assert _extract_response_list({"response": [{"a": 1}]}) == [{"a": 1}]
    assert _extract_response_list({"response": "x"}) == []
    assert _safe_int("3") == 3
    assert _safe_int("bad") is None


def test_get_fixture_odds_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "response": [
            {
                "bookmakers": [
                    {
                        "id": 1,
                        "name": "BM",
                        "bets": [
                            {
                                "name": "Match Winner",
                                "values": [
                                    {"value": "Home", "odd": "1.80"},
                                    {"value": "Draw", "odd": "3.50"},
                                    {"value": "Away", "odd": "4.20"},
                                ],
                            }
                        ],
                    }
                ]
            }
        ]
    }
    client = DummyClient(payload)
    svc = OddsService(client=client, ttl_seconds=60)
    odds = svc.get_fixture_odds(10)
    assert len(odds) == 1
    assert odds[0].home == Decimal("1.80")
    assert svc.get_cached_bookmaker_id("BM") == 1
    # cached payload -> client not called again for same fixture
    svc.get_fixture_odds(10)
    assert client.calls == 1


def test_get_fixture_bookmakers_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    payload_empty = {"response": []}
    payload_full = {
        "response": [
            {
                "bookmakers": [
                    {
                        "id": 2,
                        "name": "RetryBM",
                        "bets": [{"name": "Match Winner", "values": []}],
                    }
                ]
            }
        ]
    }
    calls = {"count": 0}

    class RetryClient:
        def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
            calls["count"] += 1
            return payload_empty if calls["count"] == 1 else payload_full

    svc = OddsService(client=RetryClient(), ttl_seconds=5)
    # Force refresh on second call via internal logic (first empty => retry -> gets full)
    bms = svc.get_fixture_bookmakers(20)
    assert bms == {2: "RetryBM"}
    assert calls["count"] == 2


def test_negative_cache_when_no_1x2(monkeypatch: pytest.MonkeyPatch) -> None:
    payload_no_1x2 = {
        "response": [
            {"bookmakers": [{"id": 3, "name": "No1X2", "bets": [{"name": "Other", "values": []}]}]}
        ]
    }

    class C:
        def __init__(self):
            self.calls = 0

        def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
            self.calls += 1
            return payload_no_1x2

    c = C()
    svc = OddsService(client=c, ttl_seconds=5)
    out1 = svc.get_fixture_odds(30)
    out2 = svc.get_fixture_odds(30)  # hits negative cache
    assert out1 == [] and out2 == []
    assert c.calls == 1


def test_invalid_odd_values_are_skipped() -> None:
    payload = {
        "response": [
            {
                "bookmakers": [
                    {
                        "id": 9,
                        "name": "BM",
                        "bets": [
                            {
                                "name": "1X2",
                                "values": [
                                    {"value": "Home", "odd": "bad"},  # invalid decimal -> skipped
                                    {"value": "Draw", "odd": "3.0"},
                                    {"value": "Away", "odd": "4.0"},
                                ],
                            }
                        ],
                    }
                ]
            }
        ]
    }
    svc = OddsService(client=DummyClient(payload), ttl_seconds=5)
    odds = svc.get_fixture_odds(55)
    # Missing home odd -> no record
    assert odds == []


def test_get_fixture_bookmakers_sleep_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    payload_empty = {"response": []}
    payload_full = {
        "response": [
            {"bookmakers": [{"id": 7, "name": "Seven", "bets": []}]},
        ]
    }
    calls = {"n": 0}

    class C:
        def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
            calls["n"] += 1
            return payload_empty if calls["n"] == 1 else payload_full

    svc = OddsService(client=C(), ttl_seconds=5)
    monkeypatch.setattr(
        odds_service.time, "sleep", lambda s: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    bms = svc.get_fixture_bookmakers(77)
    assert bms == {7: "Seven"}
    assert calls["n"] == 2


def test_get_payload_exception_sets_negative_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    class C:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
            self.calls += 1
            return {"response": []}

    svc = OddsService(client=C(), ttl_seconds=5)
    monkeypatch.setattr(
        odds_service,
        "_extract_response_list",
        lambda p: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    payload = svc._get_payload_for_fixture(888)
    assert payload == {"response": []}
    assert svc._fixture_negative_cache.get(888) is True


def test_default_client_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    created = {"n": 0}

    class _Client:
        def __init__(self) -> None:
            created["n"] += 1

        def get(self, path: str, params):
            return {"response": []}

    def fake_init(self, client=None, ttl_seconds=24 * 60 * 60):
        setattr(self, "_client", _Client())
        from src.infrastructure.ttl_cache import TTLCache

        self._bookmaker_cache = TTLCache(ttl_seconds)
        self._fixture_payload_cache = TTLCache(min(ttl_seconds, 300))
        self._fixture_negative_cache = TTLCache(30.0)

    monkeypatch.setattr(OddsService, "__init__", fake_init)
    svc = OddsService()  # type: ignore[call-arg]
    assert created["n"] == 1
    assert svc.get_fixture_odds(1) == []


def test_default_client_branch_import(monkeypatch: pytest.MonkeyPatch) -> None:
    created = {"n": 0}

    class _Client:
        def __init__(self) -> None:
            created["n"] += 1

        def get(
            self, path: str, params: Mapping[str, Any] | None = None
        ) -> Any:  # pragma: no cover - simple stub
            return {"response": [{"bookmakers": []}]}

    # Force import inside __init__ to use stub without touching real network client
    monkeypatch.setitem(
        sys.modules,
        "src.infrastructure.api_football_client",
        SimpleNamespace(APIFootballClient=_Client),
    )

    svc = OddsService()
    mapping = svc.get_fixture_bookmakers(1)
    assert created["n"] == 1
    assert mapping == {}


def test_get_fixture_bookmakers_populates_mapping() -> None:
    payload = {"response": [{"bookmakers": [{"id": 5, "name": "Five", "bets": []}]}]}
    svc = OddsService(client=DummyClient(payload), ttl_seconds=5)
    out = svc.get_fixture_bookmakers(55)
    assert out == {5: "Five"}


def test_negative_cache_set_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = OddsService(client=DummyClient({"response": []}), ttl_seconds=5)

    def boom(*args, **kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(svc._fixture_negative_cache, "set", boom)
    monkeypatch.setattr(
        odds_service,
        "_extract_response_list",
        lambda payload: (_ for _ in ()).throw(RuntimeError("explode")),
    )
    payload = svc._get_payload_for_fixture(999)
    assert payload == {"response": []}


# ---- merged original odds tests ----


class _FakeClient:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.calls: list[Mapping[str, Any]] = []

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        assert path == "odds"
        self.calls.append(params or {})
        return self.payload


def _payload(bookmakers: list[dict[str, Any]]) -> dict[str, Any]:
    return {"response": [{"bookmakers": bookmakers}], "paging": {"current": 1, "total": 1}}


def test_odds_1x2_normalization_and_filter_original() -> None:
    bookmakers = [
        {
            "id": 8,
            "name": "bet365",
            "bets": [
                {
                    "name": "Match Winner",
                    "values": [
                        {"value": "Home", "odd": "1.80"},
                        {"value": "Draw", "odd": "3.60"},
                        {"value": "Away", "odd": "4.50"},
                    ],
                },
                {"name": "Over/Under", "values": []},
            ],
        },
        {
            "id": 12,
            "name": "Other",
            "bets": [
                {
                    "name": "1X2",
                    "values": [
                        {"value": "Home", "odd": "1.00"},
                        {"value": "Draw", "odd": "3.10"},
                        {"value": "Away", "odd": "2.90"},
                    ],
                }
            ],
        },
    ]
    svc = OddsService(client=_FakeClient(_payload(bookmakers)))

    res = svc.get_fixture_odds(1234)
    assert len(res) == 1
    o = res[0]
    assert o.bookmaker_id == 8
    assert o.fixture_id == 1234
    assert o.home == Decimal("1.80") and o.draw == Decimal("3.60") and o.away == Decimal("4.50")


def test_bookmaker_cache_populated_original() -> None:
    bookmakers = [
        {
            "id": 99,
            "name": "CoolBook",
            "bets": [
                {
                    "name": "Match Winner",
                    "values": [
                        {"value": "Home", "odd": "2.00"},
                        {"value": "Draw", "odd": "3.20"},
                        {"value": "Away", "odd": "3.70"},
                    ],
                }
            ],
        }
    ]
    svc = OddsService(client=_FakeClient(_payload(bookmakers)))
    _ = svc.get_fixture_odds(4321)
    assert svc.get_cached_bookmaker_id("CoolBook") == 99
