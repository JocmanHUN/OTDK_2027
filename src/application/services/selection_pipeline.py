from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Protocol, TypedDict

logger = logging.getLogger(__name__)


class _LeaguesSvc(Protocol):
    def get_current_leagues(self) -> list[Any]: ...


class _FixturesSvc(Protocol):
    def get_daily_fixtures(
        self,
        for_date: Any,
        *,
        league_id: int | None = None,
        season: int | None = None,
        tz_name: str = "Europe/Budapest",
    ) -> list[Any]: ...


class _OddsSvc(Protocol):
    def get_fixture_odds(self, fixture_id: int) -> list[Any]: ...


class LeagueSeason(TypedDict):
    league_id: int
    season_year: int


class FixtureShort(TypedDict):
    fixture_id: int
    league_id: int
    season_year: int
    home_id: int | None
    away_id: int | None
    status: str | None


@dataclass
class SelectionPipeline:
    leagues_svc: _LeaguesSvc
    fixtures_svc: _FixturesSvc
    odds_svc: _OddsSvc

    @classmethod
    def default(cls) -> "SelectionPipeline":
        # Lazy imports to avoid API key need during tests
        from src.application.services.fixtures_service import FixturesService
        from src.application.services.leagues_service import LeaguesService
        from src.application.services.odds_service import OddsService

        return cls(LeaguesService(), FixturesService(), OddsService())

    # 1) List supported leagues
    def list_supported_leagues(self) -> list[LeagueSeason]:
        leagues = self.leagues_svc.get_current_leagues()
        out: list[LeagueSeason] = []
        for item in leagues:
            try:
                out.append(
                    LeagueSeason(
                        league_id=int(item.get("league_id")),
                        season_year=int(item.get("season_year")),
                    )
                )
            except Exception as exc:
                logger.warning("Skip league due to error: %s", exc)
        return out

    # 2) List daily matches for given leagues
    def list_daily_matches(
        self,
        for_date: str,
        leagues: Iterable[LeagueSeason],
        *,
        tz_name: str = "Europe/Budapest",
    ) -> list[FixtureShort]:
        fixtures: list[FixtureShort] = []
        for ls in leagues:
            try:
                rows = self.fixtures_svc.get_daily_fixtures(
                    for_date, league_id=ls["league_id"], season=ls["season_year"], tz_name=tz_name
                )
            except Exception as exc:
                logger.warning(
                    "Skip league %s season %s due to fixtures error: %s",
                    ls["league_id"],
                    ls["season_year"],
                    exc,
                )
                continue

            for r in rows:
                fx_id = _to_int_opt(r.get("fixture_id")) if isinstance(r, Mapping) else None
                if fx_id is None and isinstance(r, Mapping):
                    fx = r.get("fixture")
                    if isinstance(fx, Mapping):
                        fx_id = _to_int_opt(fx.get("id"))
                if fx_id is None:
                    continue
                teams = r.get("teams") if isinstance(r, Mapping) else None
                hid = (
                    teams.get("home", {}).get("id")
                    if isinstance(teams, Mapping)
                    else (r.get("home_id") if isinstance(r, Mapping) else None)
                )
                aid = (
                    teams.get("away", {}).get("id")
                    if isinstance(teams, Mapping)
                    else (r.get("away_id") if isinstance(r, Mapping) else None)
                )
                home_id = _to_int_opt(hid)
                away_id = _to_int_opt(aid)
                # status short if available (NS/1H/HT/2H/FT)
                status = None
                if isinstance(r, Mapping):
                    fx = r.get("fixture")
                    if isinstance(fx, Mapping):
                        st = fx.get("status")
                        if isinstance(st, Mapping):
                            status = st.get("short")
                fixtures.append(
                    FixtureShort(
                        fixture_id=int(fx_id),
                        league_id=ls["league_id"],
                        season_year=ls["season_year"],
                        home_id=home_id,
                        away_id=away_id,
                        status=status if isinstance(status, str) else None,
                    )
                )
        return fixtures

    # 3) Filter matches that have 1X2 odds available
    def filter_matches_with_1x2_odds(self, fixtures: Iterable[FixtureShort]) -> list[FixtureShort]:
        out: list[FixtureShort] = []
        for f in fixtures:
            try:
                odds = self.odds_svc.get_fixture_odds(int(f["fixture_id"]))
                if odds:
                    out.append(f)
            except Exception as exc:
                logger.warning("Skip fixture %s due to odds error: %s", f.get("fixture_id"), exc)
        return out


def _to_int_opt(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
