from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Mapping

from src.application.services.odds_service import OddsService


class _FakeClient:
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        self.calls: List[Mapping[str, Any]] = []

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        assert path == "odds"
        self.calls.append(params or {})
        return self.payload


def _payload(bookmakers: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"response": [{"bookmakers": bookmakers}], "paging": {"current": 1, "total": 1}}


def test_odds_1x2_normalization_and_filter() -> None:
    # Two bookmakers; one valid 1X2, one with invalid odds (<1.01) to be skipped
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
                {"name": "Over/Under", "values": []},  # ignored
            ],
        },
        {
            "id": 12,
            "name": "Other",
            "bets": [
                {
                    "name": "1X2",
                    "values": [
                        {"value": "Home", "odd": "1.00"},  # invalid -> skip whole bookmaker entry
                        {"value": "Draw", "odd": "3.10"},
                        {"value": "Away", "odd": "2.90"},
                    ],
                }
            ],
        },
    ]
    client = _FakeClient(_payload(bookmakers))
    svc = OddsService(client=client)

    res = svc.get_fixture_odds(1234)
    assert len(res) == 1
    o = res[0]
    assert o.bookmaker_id == 8
    assert o.fixture_id == 1234
    assert o.home == Decimal("1.80") and o.draw == Decimal("3.60") and o.away == Decimal("4.50")


def test_bookmaker_cache_populated() -> None:
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
