# mypy: ignore-errors

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from src.application.services.fixtures_service import (
    FixturesService,
    _extract_response_list,
    _extract_total_pages,
    _has_page_error,
    _map_fixture,
    _normalize_date_string,
    _safe_int,
    _to_utc_datetime,
)


class DummyClient:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self.calls: list[Mapping[str, Any] | None] = []

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        idx = len(self.calls)
        self.calls.append(params or {})
        if idx < len(self.pages):
            return self.pages[idx]
        return {"response": [], "errors": {"page": "no more"}}


def test_helpers() -> None:
    assert _normalize_date_string("2024-01-01") == "2024-01-01"
    assert _normalize_date_string(datetime(2024, 1, 2)) == "2024-01-02"
    assert _normalize_date_string(date(2024, 1, 3)) == "2024-01-03"
    assert _extract_response_list({"response": [1, 2]}) == [1, 2]
    assert _extract_response_list({"response": None}) == []
    assert _extract_total_pages({"paging": {"total": "3"}}) == 3
    assert _extract_total_pages({"paging": {"total": None}}) is None
    assert _extract_total_pages({"paging": {"total": "bad"}}) is None
    assert _has_page_error({"errors": {"page": "bad page"}}) is True
    assert _safe_int("5", default=None) == 5
    assert _safe_int("bad", default=None) is None
    dt = _to_utc_datetime("2024-01-01T12:00:00Z")
    assert dt.tzinfo is not None and dt.utcoffset().total_seconds() == 0


def test_fetch_all_pages_with_paging() -> None:
    pages = [
        {"response": [{"fixture": {"id": 1}}], "paging": {"total": 2}},
        {"response": [{"fixture": {"id": 2}}], "paging": {"total": 2}},
    ]
    svc = FixturesService(client=DummyClient(pages))
    out = svc.get_daily_fixtures("2024-01-01")
    ids = [f["fixture_id"] for f in out]
    assert ids == [1, 2]


def test_fetch_all_pages_probe_until_error() -> None:
    # No paging info, but second page returns page error -> stop
    pages = [
        {"response": [{"fixture": {"id": 1}}]},
        {"response": [], "errors": {"page": "invalid"}},
    ]
    svc = FixturesService(client=DummyClient(pages))
    out = svc.get_daily_fixtures("2024-01-01")
    assert len(out) == 1 and out[0]["fixture_id"] == 1


def test_map_fixture_fields() -> None:
    src = {
        "fixture": {
            "id": "10",
            "date": "2024-01-01T12:00:00Z",
            "status": {"short": "FT"},
        },
        "league": {"id": "20", "season": "2024"},
        "teams": {"home": {"id": "1", "name": "H"}, "away": {"id": "2", "name": "A"}},
    }
    mapped = _map_fixture(src)
    assert mapped["fixture_id"] == 10
    assert mapped["league_id"] == 20
    assert mapped["home_name"] == "H"
    assert mapped["status"] == "FT"


def test_default_client_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    created = {"n": 0}

    class _Client:
        def __init__(self) -> None:
            created["n"] += 1

        def get(
            self, path: str, params: Mapping[str, Any] | None = None
        ) -> Any:  # pragma: no cover - stub
            return {"response": [], "paging": {"total": 1}}

    monkeypatch.setitem(
        sys.modules,
        "src.infrastructure.api_football_client",
        SimpleNamespace(APIFootballClient=_Client),
    )
    svc = FixturesService()
    assert svc.get_daily_fixtures("2024-01-01") == []
    assert created["n"] == 1


def test_fetch_all_pages_breaks_on_empty_batch() -> None:
    pages = [
        {"response": [{"fixture": {"id": 1}}]},
        {"response": []},  # triggers break when batch empty
    ]
    svc = FixturesService(client=DummyClient(pages))
    rows = svc.get_daily_fixtures("2024-01-01")
    assert [r["fixture_id"] for r in rows] == [1]


def test_fetch_all_pages_with_total_and_page_error() -> None:
    pages = [
        {"response": [{"fixture": {"id": 1}}], "paging": {"total": 3}},
        {"response": [], "errors": {"page": "stop"}},  # breaks inside total_pages loop
        {"response": [{"fixture": {"id": 3}}]},
    ]
    svc = FixturesService(client=DummyClient(pages))
    rows = svc.get_daily_fixtures("2024-01-01")
    assert [r["fixture_id"] for r in rows] == [1]


def test_extract_total_pages_and_page_error_edges() -> None:
    assert _extract_total_pages("bad") is None
    assert _has_page_error("oops") is False
    # mapping error value branch
    assert _has_page_error({"errors": {"page": {"msg": "err"}}}) is True
    # non-empty non-str/list/mapping value
    assert _has_page_error({"errors": {"page": 0}}) is True


def test_to_utc_datetime_fallback_with_fractional_noise() -> None:
    dt = _to_utc_datetime("2024-01-01T00:00:00.000bad")
    assert dt.tzinfo is not None and dt.utcoffset().total_seconds() == 0


# ---- merged edge coverage tests ----


def test_normalize_and_to_utc_datetime_fallbacks() -> None:
    assert _normalize_date_string(date(2024, 1, 1)) == "2024-01-01"
    assert _normalize_date_string(datetime(2024, 1, 1, tzinfo=timezone.utc)) == "2024-01-01"
    dt = _to_utc_datetime("2024-01-01T12:00:00")
    assert dt.tzinfo is not None and dt.utcoffset().total_seconds() == 0


def test_extract_total_pages_invalid_types() -> None:
    assert _extract_total_pages({"paging": {"total": -1}}) is None
    assert _extract_total_pages({"paging": {"total": "bad"}}) is None
    assert _extract_total_pages({"paging": None}) is None


def test_fetch_pages_until_page_error() -> None:
    pages = [
        {"response": [{"fixture": {"id": 1}}], "paging": {"total": None}, "errors": {}},
        {"response": [{"fixture": {"id": 2}}], "errors": {"page": []}},
        {"response": [], "errors": {"page": "stop"}},
        {"response": [{"fixture": {"id": 3}}]},
    ]
    svc = FixturesService(client=DummyClient(pages))
    out = svc.get_daily_fixtures("2024-01-01")
    ids = [f["fixture_id"] for f in out]
    assert ids == [1, 2]


def test_map_fixture_defaults() -> None:
    mapped = _map_fixture({"fixture": {}, "league": {}, "teams": {}})
    assert mapped["fixture_id"] == 0
    assert mapped["league_id"] == 0
    assert mapped["season"] is None
    assert mapped["status"] is None


# ---- merged original fixtures tests ----
class _FakeClient:
    def __init__(self, pages: list[dict[str, Any]]):
        self.pages = pages
        self.calls: list[Mapping[str, Any]] = []

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        assert path == "fixtures"
        self.calls.append(params or {})
        idx = min(len(self.calls) - 1, len(self.pages) - 1)
        return self.pages[idx]


def _page(payload_items: list[dict[str, Any]], current: int, total: int) -> dict[str, Any]:
    return {"response": payload_items, "paging": {"current": current, "total": total}}


def test_pagination_and_param_forwarding_original() -> None:
    items1 = [
        {
            "fixture": {"id": 101, "date": "2025-03-10T18:00:00+02:00", "status": {"short": "NS"}},
            "league": {"id": 39, "season": 2025},
            "teams": {"home": {"id": 10, "name": "A"}, "away": {"id": 20, "name": "B"}},
        }
    ]
    items2 = [
        {
            "fixture": {"id": 102, "date": "2025-03-10T20:30:00+02:00", "status": {"short": "NS"}},
            "league": {"id": 39, "season": 2025},
            "teams": {"home": {"id": 30, "name": "C"}, "away": {"id": 40, "name": "D"}},
        }
    ]
    client = _FakeClient(
        [
            _page(items1, current=1, total=2),
            _page(items2, current=2, total=2),
        ]
    )

    svc = FixturesService(client=client)
    rows = svc.get_daily_fixtures("2025-03-10", league_id=39, season=2025)

    assert len(rows) == 2
    assert rows[0]["fixture_id"] == 101 and rows[1]["fixture_id"] == 102
    assert "page" not in client.calls[0]
    assert client.calls[0]["timezone"] == "Europe/Budapest"
    assert str(client.calls[0]["league"]) == "39" and str(client.calls[0]["season"]) == "2025"
    assert client.calls[1]["page"] in (2, "2")


def test_utc_conversion_original() -> None:
    client = _FakeClient(
        [
            _page(
                [
                    {
                        "fixture": {
                            "id": 7,
                            "date": "2025-02-01T18:00:00+02:00",
                            "status": {"short": "NS"},
                        },
                        "league": {"id": 39, "season": 2025},
                        "teams": {"home": {"id": 1, "name": "H"}, "away": {"id": 2, "name": "A"}},
                    }
                ],
                1,
                1,
            )
        ]
    )

    svc = FixturesService(client=client)
    rows = svc.get_daily_fixtures("2025-02-01")
    dt = rows[0]["date_utc"]
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc
    assert dt.hour == 16 and dt.minute == 0


def test_empty_response_original() -> None:
    client = _FakeClient([_page([], 1, 1)])
    svc = FixturesService(client=client)
    rows = svc.get_daily_fixtures("2025-03-10")
    assert rows == []


def test_page_probe_aborts_when_api_rejects_original() -> None:
    first_payload = {
        "response": [
            {
                "fixture": {
                    "id": 555,
                    "date": "2025-03-10T12:00:00+00:00",
                    "status": {"short": "NS"},
                },
                "league": {"id": 9, "season": 2025},
                "teams": {"home": {"id": 1, "name": "Foo"}, "away": {"id": 2, "name": "Bar"}},
            }
        ]
    }
    error_payload = {
        "errors": {"page": "The Page field do not exist."},
        "response": [],
    }
    client = _FakeClient([first_payload, error_payload])
    svc = FixturesService(client=client)
    rows = svc.get_daily_fixtures("2025-03-10")

    assert len(rows) == 1
    assert len(client.calls) == 2
    assert "page" not in client.calls[0]
    assert client.calls[1]["page"] in (2, "2")
