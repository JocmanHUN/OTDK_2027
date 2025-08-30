from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Protocol, TypedDict

from src.infrastructure.ttl_cache import TTLCache

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    pass


class LeagueSeason(TypedDict):
    league_id: int
    league_name: str
    country_name: str | None
    season_year: int
    has_odds: bool
    has_stats: bool


class _ClientProto(Protocol):
    def get(self, path: str, params: Mapping[str, Any] | None = None) -> object: ...


class LeaguesService:
    """Service to fetch leagues and seasons with coverage filtering and caching.

    - Uses API-FOOTBALL ``/leagues`` endpoint with either ``current=true`` or ``season=YYYY``.
    - Filters to only leagues/seasons where odds and statistics coverage are available.
    - Caches results in-memory for ``ttl_seconds`` (default 24h).
    """

    def __init__(
        self,
        client: Optional[_ClientProto] = None,
        ttl_seconds: float = 24 * 60 * 60,
    ) -> None:
        self._client: _ClientProto
        if client is None:
            # Lazy import to avoid requiring API credentials during tests that inject a fake client
            from src.infrastructure.api_football_client import APIFootballClient as _Client

            self._client = _Client()
        else:
            self._client = client
        self._cache_current = TTLCache[str, list[LeagueSeason]](ttl_seconds)
        self._cache_season = TTLCache[int, list[LeagueSeason]](ttl_seconds)

    def get_current_leagues(self) -> list[LeagueSeason]:
        cached = self._cache_current.get("current")
        if cached is not None:
            return cached

        payload = self._client.get("leagues", params={"current": "true"})
        items = _parse_leagues_response(payload)
        # Filter by current season only
        result: list[LeagueSeason] = []
        for item in items:
            league = item.get("league") or {}
            country = item.get("country") or {}
            for s in item.get("seasons", []) or []:
                if not s.get("current"):
                    continue
                cov = s.get("coverage") or {}
                if _has_odds_and_stats(cov):
                    result.append(
                        LeagueSeason(
                            league_id=_safe_int(league.get("id")),
                            league_name=str(league.get("name")),
                            country_name=(
                                country.get("name") if country.get("name") is not None else None
                            ),
                            season_year=_safe_int(s.get("year")),
                            has_odds=True,
                            has_stats=True,
                        )
                    )
        self._cache_current.set("current", result)
        return result

    def get_leagues_for_season(self, year: int) -> list[LeagueSeason]:
        cached = self._cache_season.get(year)
        if cached is not None:
            return cached

        payload = self._client.get("leagues", params={"season": year})
        items = _parse_leagues_response(payload)
        result: list[LeagueSeason] = []
        for item in items:
            league = item.get("league") or {}
            country = item.get("country") or {}
            for s in item.get("seasons", []) or []:
                if _safe_int(s.get("year")) != int(year):
                    continue
                cov = s.get("coverage") or {}
                if _has_odds_and_stats(cov):
                    result.append(
                        LeagueSeason(
                            league_id=_safe_int(league.get("id")),
                            league_name=str(league.get("name")),
                            country_name=(
                                country.get("name") if country.get("name") is not None else None
                            ),
                            season_year=_safe_int(s.get("year")),
                            has_odds=True,
                            has_stats=True,
                        )
                    )
        self._cache_season.set(year, result)
        return result


def _parse_leagues_response(payload: Any) -> list[Mapping[str, Any]]:
    """Extract the API's response list from payload.

    API-FOOTBALL wraps results under ``response``.
    """
    if isinstance(payload, dict) and "response" in payload:
        data = payload.get("response")
        if isinstance(data, list):
            return data
    # If unexpected shape, return empty list to follow skip & log principle (logging elsewhere)
    return []


def _has_odds_and_stats(coverage: Mapping[str, Any]) -> bool:
    """Return True if both odds and statistics coverage are available.

    Heuristic based on API-FOOTBALL structure:
    - ``coverage.get("odds")`` must be truthy
    - ``coverage.get("fixtures", {}).get("statistics")`` must be truthy
    """
    odds_ok = bool(coverage.get("odds"))
    fixtures = coverage.get("fixtures") or {}
    # API-FOOTBALL uses either "statistics" or "statistics_fixtures" in fixtures coverage
    stats_ok = bool(fixtures.get("statistics") or fixtures.get("statistics_fixtures"))
    return odds_ok and stats_ok


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
