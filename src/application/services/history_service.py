from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Protocol


class _ClientProto(Protocol):
    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any: ...


@dataclass(frozen=True)
class TeamAverages:
    matches_home: int
    matches_away: int
    goals_for_home_avg: float
    goals_for_away_avg: float
    goals_against_home_avg: float
    goals_against_away_avg: float


@dataclass(frozen=True)
class EloInput:
    home_team_id: int
    away_team_id: int
    init_rating_home: float
    init_rating_away: float
    home_advantage: float


class HistoryService:
    """Provides historical data and derived inputs for modeling.

    Endpoints used:
    - /fixtures/headtohead (h2h: "{home}-{away}")
    - /teams/statistics (team, league, season)

    Outputs:
    - Clean head-to-head match list
    - Team home/away goal averages (for/against)
    - Simple Poisson means and Elo input scaffold
    """

    def __init__(self, client: Optional[_ClientProto] = None) -> None:
        if client is None:
            from src.infrastructure.api_football_client import APIFootballClient as _Client

            self._client: _ClientProto = _Client()
        else:
            self._client = client

    # --------------------------- Head-to-head ---------------------------
    def get_head_to_head(
        self, home_team_id: int, away_team_id: int, last: int = 20
    ) -> list[dict[str, Any]]:
        params = {"h2h": f"{home_team_id}-{away_team_id}", "last": str(last)}
        payload = self._client.get("fixtures/headtohead", params)
        rows: list[dict[str, Any]] = []

        if isinstance(payload, Mapping):
            for item in payload.get("response", []) or []:
                fx = item.get("fixture") or {}
                teams = item.get("teams") or {}
                home = teams.get("home") or {}
                away = teams.get("away") or {}
                goals = item.get("goals") or {}
                # Determine winner label
                if home.get("winner") is True:
                    result = "home"
                elif away.get("winner") is True:
                    result = "away"
                else:
                    result = "draw"

                date_str = fx.get("date")
                dt = _to_utc(date_str) if isinstance(date_str, str) else datetime.now(timezone.utc)
                rows.append(
                    {
                        "date_utc": dt,
                        "home_id": _safe_int(home.get("id")),
                        "away_id": _safe_int(away.get("id")),
                        "home_goals": _safe_int(goals.get("home")),
                        "away_goals": _safe_int(goals.get("away")),
                        "result": result,
                    }
                )
        return rows

    # --------------------------- Team statistics ---------------------------
    def get_team_averages(self, team_id: int, league_id: int, season: int) -> TeamAverages:
        params = {"team": str(team_id), "league": str(league_id), "season": str(season)}
        payload = self._client.get("teams/statistics", params)

        def _get(path: list[str], default: Any = None) -> Any:
            cur: Any = payload
            for key in path:
                if not isinstance(cur, Mapping):
                    return default
                cur = cur.get(key)
            return cur if cur is not None else default

        # Matches played
        matches_home = int(_safe_float(_get(["response", "fixtures", "played", "home"], 0.0)))
        matches_away = int(_safe_float(_get(["response", "fixtures", "played", "away"], 0.0)))

        # Goals averages are often strings like "1.6"; parse robustly
        gf_home = _safe_float(_get(["response", "goals", "for", "average", "home"], 0.0))
        gf_away = _safe_float(_get(["response", "goals", "for", "average", "away"], 0.0))
        ga_home = _safe_float(_get(["response", "goals", "against", "average", "home"], 0.0))
        ga_away = _safe_float(_get(["response", "goals", "against", "average", "away"], 0.0))

        return TeamAverages(
            matches_home=matches_home,
            matches_away=matches_away,
            goals_for_home_avg=gf_home,
            goals_for_away_avg=gf_away,
            goals_against_home_avg=ga_home,
            goals_against_away_avg=ga_away,
        )

    # --------------------------- Derived inputs ---------------------------
    def simple_poisson_means(self, home: TeamAverages, away: TeamAverages) -> tuple[float, float]:
        """Compute simple expected goals using home/away averages.

        Uses the mean of (home GF at home) and (away GA away) for the home team,
        and the mean of (away GF away) and (home GA home) for the away team.
        """

        mu_home = (home.goals_for_home_avg + away.goals_against_away_avg) / 2.0
        mu_away = (away.goals_for_away_avg + home.goals_against_home_avg) / 2.0
        return (mu_home, mu_away)

    def elo_input(
        self,
        home_team_id: int,
        away_team_id: int,
        *,
        init_rating: float = 1500.0,
        home_advantage: float = 100.0,
    ) -> EloInput:
        return EloInput(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            init_rating_home=float(init_rating),
            init_rating_away=float(init_rating),
            home_advantage=float(home_advantage),
        )

    # --------------------------- Recent team fixtures + stats ---------------------------
    def get_recent_team_stats(
        self,
        team_id: int,
        league_id: int,
        season: int,
        last: int,
        *,
        only_finished: bool = True,
    ) -> list[dict[str, Any]]:
        """Return the last N league fixtures for a team, backfilling previous seasons if needed.

        Each item includes: fixture_id, date_utc, home_away, opponent_id, goals_for, goals_against,
        and a 'stats' dict (shots on target, possession, etc.) when available.
        """
        needed = int(last)
        cur_season = int(season)
        collected: list[dict[str, Any]] = []

        while needed > 0 and cur_season >= 2000:  # basic guard
            fixtures = self._fetch_team_fixtures_for_season(
                team_id, league_id, cur_season, only_finished=only_finished
            )
            # Sort by date desc
            fixtures.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
            for fx in fixtures:
                if needed <= 0:
                    break
                # Basic fields
                f_id = fx.get("id")
                ts = fx.get("timestamp")
                dt = (
                    datetime.fromtimestamp(int(ts), tz=timezone.utc)
                    if ts
                    else _to_utc(str(fx.get("date", "")))
                )
                teams = fx.get("teams") or {}
                home = teams.get("home") or {}
                away = teams.get("away") or {}
                is_home = int(home.get("id") or -1) == int(team_id)
                opponent_id = int((away if is_home else home).get("id") or 0)
                goals = fx.get("goals") or {}
                gf = int(goals.get("home") or 0) if is_home else int(goals.get("away") or 0)
                ga = int(goals.get("away") or 0) if is_home else int(goals.get("home") or 0)

                stats = self.get_fixture_statistics(int(f_id)) if f_id else {}
                team_stats = stats.get(int(team_id)) or {}

                collected.append(
                    {
                        "fixture_id": int(f_id) if f_id is not None else None,
                        "date_utc": dt,
                        "home_away": "H" if is_home else "A",
                        "opponent_id": opponent_id,
                        "goals_for": gf,
                        "goals_against": ga,
                        # Stats for the requested team
                        "stats": team_stats,
                        # All teams' stats in this fixture {team_id: {stat: value}}
                        "all_stats": stats,
                    }
                )
                needed -= 1

            cur_season -= 1

        return collected[:last]

    def _fetch_team_fixtures_for_season(
        self, team_id: int, league_id: int, season: int, *, only_finished: bool = True
    ) -> list[dict[str, Any]]:
        params = {
            "team": str(team_id),
            "league": str(league_id),
            "season": str(season),
        }
        if only_finished:
            params["status"] = "FT"
        payload = self._client.get("fixtures", params)
        out: list[dict[str, Any]] = []
        if isinstance(payload, Mapping):
            for item in payload.get("response", []) or []:
                fx = item.get("fixture") or {}
                status = (
                    (fx.get("status") or {}).get("short")
                    if isinstance(fx.get("status"), Mapping)
                    else None
                )
                # In case API ignores status filter, enforce it here too
                if only_finished and status != "FT":
                    continue
                teams = item.get("teams") or {}
                goals = item.get("goals") or {}
                out.append(
                    {
                        "id": _safe_int(fx.get("id")),
                        "timestamp": _safe_int(fx.get("timestamp")),
                        "date": fx.get("date"),
                        "teams": teams,
                        "goals": goals,
                        "status": status,
                    }
                )
        return out

    def get_fixture_statistics(self, fixture_id: int) -> dict[int, dict[str, float]]:
        """Return normalized per-team statistics for a given fixture.

        Output mapping: { team_id: { stat_name: value_float, ... }, ... }
        """
        payload = self._client.get("fixtures/statistics", {"fixture": str(int(fixture_id))})
        result: dict[int, dict[str, float]] = {}
        if isinstance(payload, Mapping):
            for item in payload.get("response", []) or []:
                team = item.get("team") or {}
                t_id = _safe_int(team.get("id"))
                stats_arr = item.get("statistics") or []
                if t_id is None:
                    continue
                bucket: dict[str, float] = {}
                for s in stats_arr:
                    name = str(s.get("type") or "").strip().lower()
                    val = s.get("value")
                    norm = _normalize_stat_value(val)
                    if norm is not None and name:
                        bucket[name] = norm
                result[int(t_id)] = bucket
        return result


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_utc(iso_str: str) -> datetime:
    s = iso_str.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.fromisoformat(s.split(".")[0]) if "." in s else datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_stat_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    # Percent strings like "55%"
    if isinstance(value, str) and value.strip().endswith("%"):
        try:
            return float(value.strip().rstrip("%"))
        except ValueError:
            return None
    # General numbers
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
