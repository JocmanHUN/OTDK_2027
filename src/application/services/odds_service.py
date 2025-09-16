from __future__ import annotations

import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Optional, Protocol

from src.domain.entities.odds import Odds
from src.domain.value_objects.ids import BookmakerId, FixtureId
from src.infrastructure.ttl_cache import TTLCache


class _ClientProto(Protocol):
    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any: ...


class OddsService:
    """Fetch 1X2 odds for a fixture and normalize decimal odds.

    - Calls /odds?fixture={fixture_id}
    - Extracts only the 1X2 market (aka "Match Winner", "1X2")
    - Normalizes odds to Decimal and filters invalid entries
    - Caches bookmaker id by name for convenience (24h TTL)
    """

    def __init__(
        self, client: Optional[_ClientProto] = None, *, ttl_seconds: float = 24 * 60 * 60
    ) -> None:
        # Lazy import to avoid API key requirement during tests when a fake client is injected
        if client is None:
            from src.infrastructure.api_football_client import APIFootballClient as _Client

            self._client: _ClientProto = _Client()
        else:
            self._client = client

        self._bookmaker_cache = TTLCache[str, int](ttl_seconds)
        # Cache full odds payload per fixture to avoid double /odds calls in one session
        self._fixture_payload_cache = TTLCache[int, Any](min(ttl_seconds, 5 * 60))
        # Negative cache for fixtures that currently have no 1X2 market
        self._fixture_negative_cache = TTLCache[int, bool](30.0)

    def get_fixture_odds(self, fixture_id: int | FixtureId) -> list[Odds]:
        payload = self._get_payload_for_fixture(int(fixture_id))
        items = _extract_response_list(payload)
        out: list[Odds] = []

        for item in items:
            # API-FOOTBALL structure: item['bookmakers'] is a list
            for bm in item.get("bookmakers", []) or []:
                bm_id = _safe_int(bm.get("id"))
                bm_name = bm.get("name")
                if bm_name and bm_id is not None:
                    self._bookmaker_cache.set(str(bm_name), int(bm_id))

                bets = bm.get("bets") or bm.get("markets") or []
                for bet in bets:
                    name = str(bet.get("name") or "").strip().lower()
                    if name not in {"match winner", "1x2"}:
                        continue
                    values = bet.get("values") or []
                    odds_map: dict[str, Decimal] = {}
                    for v in values:
                        label = str(v.get("value") or v.get("label") or "").strip().lower()
                        odd_str = v.get("odd")
                        try:
                            odd = Decimal(str(odd_str))
                        except (InvalidOperation, TypeError):
                            continue
                        if label in {"home", "1"}:
                            odds_map["home"] = odd
                        elif label in {"draw", "x"}:
                            odds_map["draw"] = odd
                        elif label in {"away", "2"}:
                            odds_map["away"] = odd
                    if {"home", "draw", "away"}.issubset(odds_map.keys()) and bm_id is not None:
                        try:
                            out.append(
                                Odds(
                                    fixture_id=FixtureId(int(fixture_id)),
                                    bookmaker_id=BookmakerId(int(bm_id)),
                                    collected_at_utc=datetime.now(timezone.utc),
                                    home=odds_map["home"],
                                    draw=odds_map["draw"],
                                    away=odds_map["away"],
                                )
                            )
                        except Exception:
                            # Skip invalid record per "skip & log" philosophy; logging can be added
                            continue
        return out

    def get_cached_bookmaker_id(self, name: str) -> Optional[int]:
        return self._bookmaker_cache.get(name)

    def get_fixture_bookmakers(self, fixture_id: int | FixtureId) -> dict[int, str]:
        """Return a mapping of bookmaker_id -> bookmaker_name for the fixture.

        Uses the same /odds payload but extracts identifiers and human names.
        """
        payload = self._get_payload_for_fixture(int(fixture_id))
        items = _extract_response_list(payload)
        out: dict[int, str] = {}
        for item in items:
            for bm in item.get("bookmakers", []) or []:
                bm_id = _safe_int(bm.get("id"))
                bm_name = bm.get("name")
                if bm_id is not None and isinstance(bm_name, str) and bm_name:
                    out[int(bm_id)] = bm_name
        # Soft-retry once if nothing found
        if not out:
            try:
                time.sleep(1.2)
            except Exception:
                pass
            payload = self._get_payload_for_fixture(int(fixture_id), force_refresh=True)
            items = _extract_response_list(payload)
            for item in items:
                for bm in item.get("bookmakers", []) or []:
                    bm_id = _safe_int(bm.get("id"))
                    bm_name = bm.get("name")
                    if bm_id is not None and isinstance(bm_name, str) and bm_name:
                        out[int(bm_id)] = str(bm_name)
        return out

    def _get_payload_for_fixture(self, fixture_id: int, *, force_refresh: bool = False) -> Any:
        if not force_refresh:
            cached = self._fixture_payload_cache.get(int(fixture_id))
            if cached is not None:
                return cached
            # If recently observed as having no 1X2, avoid hammering
            if self._fixture_negative_cache.get(int(fixture_id)):
                return {"response": []}
        params = {"fixture": str(int(fixture_id))}
        payload = self._client.get("odds", params)
        # Cache logic: cache only payloads that include a 1X2 market; else short negative cache
        try:
            items = _extract_response_list(payload)
            has_1x2 = False
            for item in items:
                for bm in item.get("bookmakers", []) or []:
                    bets = bm.get("bets") or bm.get("markets") or []
                    for bet in bets:
                        name = str(bet.get("name") or "").strip().lower()
                        if name in {"match winner", "1x2"}:
                            has_1x2 = True
                            break
                    if has_1x2:
                        break
            if has_1x2:
                self._fixture_payload_cache.set(int(fixture_id), payload)
            else:
                self._fixture_negative_cache.set(int(fixture_id), True)
        except Exception:
            try:
                self._fixture_negative_cache.set(int(fixture_id), True)
            except Exception:
                pass
        return payload


def _extract_response_list(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        lst = payload.get("response")
        if isinstance(lst, list):
            return lst
    return []


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
