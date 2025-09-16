from __future__ import annotations

from datetime import date as Date
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping, Optional, Protocol, TypedDict


class _ClientProto(Protocol):
    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any: ...


class Fixture(TypedDict):
    fixture_id: int
    league_id: int
    season: int | None
    date_utc: datetime
    home_id: int | None
    away_id: int | None
    home_name: str | None
    away_name: str | None
    status: str | None


if TYPE_CHECKING:  # pragma: no cover
    pass


class FixturesService:
    """Fetch daily fixtures with pagination, timezone handling, and optional filters.

    - Interprets the requested date in Europe/Budapest and passes `timezone` to API.
    - Aggregates all pages using the API `paging` object.
    - Converts returned ISO datetime strings to UTC-aware ``datetime``.
    """

    def __init__(self, client: Optional[_ClientProto] = None) -> None:
        self._client: _ClientProto
        if client is None:
            # Lazy import to avoid requiring credentials in tests that inject fakes
            from src.infrastructure.api_football_client import APIFootballClient as _Client

            self._client = _Client()
        else:
            self._client = client

    def get_daily_fixtures(
        self,
        for_date: Date | datetime | str,
        *,
        league_id: int | None = None,
        season: int | None = None,
        tz_name: str = "Europe/Budapest",
    ) -> list[Fixture]:
        day_str = _normalize_date_string(for_date)

        base_params: dict[str, Any] = {"date": str(day_str), "timezone": str(tz_name)}
        if league_id is not None:
            base_params["league"] = str(league_id)
        if season is not None:
            base_params["season"] = str(season)

        items = self._fetch_all_pages(base_params)

        return [_map_fixture(it) for it in items]

    def _fetch_all_pages(self, params: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        out: list[Mapping[str, Any]] = []

        # First call: start with page=1 to stabilize paging behavior
        first_params = dict(params)
        first_params["page"] = "1"
        payload = self._client.get("fixtures", params=first_params)
        batch = _extract_response_list(payload)
        out.extend(batch)

        # Additional pages if needed
        paging = payload.get("paging") if isinstance(payload, Mapping) else None
        total = int(paging.get("total")) if paging and paging.get("total") is not None else 1
        if total and total > 1:
            for page in range(2, total + 1):
                more_params = dict(params)
                more_params["page"] = str(page)
                payload = self._client.get("fixtures", params=more_params)
                batch = _extract_response_list(payload)
                out.extend(batch)
        else:
            # Some API responses might omit paging; probe a few pages defensively
            for page in range(2, 6):  # probe up to page 5
                more_params = dict(params)
                more_params["page"] = str(page)
                payload = self._client.get("fixtures", params=more_params)
                batch = _extract_response_list(payload)
                if not batch:
                    break
                out.extend(batch)
        return out


def _normalize_date_string(value: Date | datetime | str) -> str:
    if isinstance(value, str):
        # assume already YYYY-MM-DD
        return value
    if isinstance(value, datetime):
        return value.date().isoformat()
    return value.isoformat()


def _extract_response_list(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        lst = payload.get("response")
        if isinstance(lst, list):
            return lst
    return []


def _map_fixture(item: Mapping[str, Any]) -> Fixture:
    fx = item.get("fixture") or {}
    lg = item.get("league") or {}
    teams = item.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}

    date_str = fx.get("date")
    dt_utc = _to_utc_datetime(date_str) if isinstance(date_str, str) else datetime.now(timezone.utc)

    return Fixture(
        fixture_id=_safe_int(fx.get("id"), default=0) or 0,
        league_id=_safe_int(lg.get("id"), default=0) or 0,
        season=_safe_int(lg.get("season"), default=None),
        date_utc=dt_utc,
        home_id=_safe_int(home.get("id"), default=None),
        away_id=_safe_int(away.get("id"), default=None),
        home_name=home.get("name"),
        away_name=away.get("name"),
        status=(
            (fx.get("status") or {}).get("short") if isinstance(fx.get("status"), Mapping) else None
        ),
    )


def _to_utc_datetime(iso_str: str) -> datetime:
    # Handle "Z" suffix and timezone offsets
    s = iso_str.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Fallback: treat as naive UTC if parsing fails
        dt = datetime.fromisoformat(s.split(".")[0]) if "." in s else datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_int(value: Any, default: int | None = 0) -> int | None:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
