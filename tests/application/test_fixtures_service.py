from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping

from src.application.services.fixtures_service import FixturesService


class _FakeClient:
    def __init__(self, pages: List[Dict[str, Any]]):
        self.pages = pages
        self.calls: List[Mapping[str, Any]] = []

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        assert path == "fixtures"
        self.calls.append(params or {})
        # pages are returned in order on each call; if exhausted, repeat last
        idx = min(len(self.calls) - 1, len(self.pages) - 1)
        return self.pages[idx]


def _page(payload_items: List[Dict[str, Any]], current: int, total: int) -> Dict[str, Any]:
    return {"response": payload_items, "paging": {"current": current, "total": total}}


def test_pagination_and_param_forwarding() -> None:
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
    # First call may omit page (API prefers no 'page' on first request)
    assert ("page" not in client.calls[0]) or (client.calls[0]["page"] in (1, "1"))
    assert client.calls[0]["timezone"] == "Europe/Budapest"
    assert str(client.calls[0]["league"]) == "39" and str(client.calls[0]["season"]) == "2025"
    # Second page requested
    assert client.calls[1]["page"] in (2, "2")


def test_utc_conversion() -> None:
    # +02:00 should convert to UTC minus 2 hours
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


def test_empty_response() -> None:
    client = _FakeClient([_page([], 1, 1)])
    svc = FixturesService(client=client)
    rows = svc.get_daily_fixtures("2025-03-10")
    assert rows == []
