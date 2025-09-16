from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Protocol

from src.infrastructure.ttl_cache import TTLCache


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

    def __init__(
        self, client: Optional[_ClientProto] = None, *, cache_ttl_seconds: float = 15 * 60
    ) -> None:
        if client is None:
            from src.infrastructure.api_football_client import APIFootballClient as _Client

            self._client: _ClientProto = _Client()
        else:
            self._client = client
        # Lightweight in-memory caches to reduce repeated API calls during a session
        self._cache_team_avg = TTLCache[tuple[int, int, int], TeamAverages](cache_ttl_seconds)
        self._cache_recent_stats = TTLCache[tuple[int, int, int, int], list](cache_ttl_seconds)
        self._cache_fixture_stats = TTLCache[int, dict[int, dict[str, float]]](cache_ttl_seconds)
        self._cache_league_means = TTLCache[tuple[int, int], tuple[float, float]](cache_ttl_seconds)

    # --------------------------- Head-to-head ---------------------------
    def get_head_to_head(
        self,
        home_team_id: int,
        away_team_id: int,
        last: int = 20,
        *,
        exclude_friendlies: bool = True,
    ) -> list[dict[str, Any]]:
        params = {"h2h": f"{home_team_id}-{away_team_id}", "last": str(last)}
        payload = self._client.get("fixtures/headtohead", params)
        rows: list[dict[str, Any]] = []

        if isinstance(payload, Mapping):
            for item in payload.get("response", []) or []:
                lg = item.get("league") or {}
                lg_type = str(lg.get("type") or "").lower()
                lg_name = str(lg.get("name") or "").lower()
                if exclude_friendlies and (
                    "friendly" in lg_name or lg_type == "friendly" or lg_type == "friendlies"
                ):
                    continue
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
                        "league_id": _safe_int(lg.get("id")),
                        "league_type": lg_type or None,
                        "league_name": lg.get("name"),
                    }
                )
        return rows

    # --------------------------- Team statistics ---------------------------
    def get_team_averages(self, team_id: int, league_id: int, season: int) -> TeamAverages:
        key = (int(team_id), int(league_id), int(season))
        cached = self._cache_team_avg.get(key)
        if cached is not None:
            return cached
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

        result = TeamAverages(
            matches_home=matches_home,
            matches_away=matches_away,
            goals_for_home_avg=gf_home,
            goals_for_away_avg=gf_away,
            goals_against_home_avg=ga_home,
            goals_against_away_avg=ga_away,
        )
        try:
            self._cache_team_avg.set(key, result)
        except Exception:
            pass
        return result

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
        exclude_friendlies: bool = True,
    ) -> list[dict[str, Any]]:
        """Return the last N league fixtures for a team, backfilling previous seasons if needed.

        Each item includes: fixture_id, date_utc, home_away, opponent_id, goals_for, goals_against,
        and a 'stats' dict (shots on target, possession, etc.) when available.
        """
        needed = int(last)
        cur_season = int(season)
        collected: list[dict[str, Any]] = []

        # Cache hit for this (team, league, season, last)
        ckey = (int(team_id), int(league_id), int(season), int(last))
        cached = self._cache_recent_stats.get(ckey)
        if cached is not None:
            return cached

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
                lg = fx.get("league") or {}
                lg_type = str((lg.get("type") or "")).lower()
                lg_name = str((lg.get("name") or "")).lower()
                if exclude_friendlies and (
                    "friendly" in lg_name or lg_type in {"friendly", "friendlies"}
                ):
                    continue
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

        result = collected[:last]
        try:
            self._cache_recent_stats.set(ckey, result)
        except Exception:
            pass
        return result

    # --------------------------- Recent team scores (cheap) ---------------------------
    def get_recent_team_scores(
        self,
        team_id: int,
        league_id: int,
        season: int,
        last: int,
        *,
        only_finished: bool = True,
    ) -> list[dict[str, Any]]:
        """Return the last N league fixtures for a team with only goals and home/away flags.

        Uses only GET /fixtures (no /fixtures/statistics), so it's much cheaper.
        Each row: {fixture_id, date_utc, home_away: 'H'|'A', goals_for, goals_against}
        """
        needed = int(last)
        cur_season = int(season)
        collected: list[dict[str, Any]] = []

        while needed > 0 and cur_season >= 2000:
            fixtures = self._fetch_team_fixtures_for_season(
                team_id, league_id, cur_season, only_finished=only_finished
            )
            fixtures.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
            for fx in fixtures:
                if needed <= 0:
                    break
                f_id = fx.get("id")
                ts = fx.get("timestamp")
                dt = (
                    datetime.fromtimestamp(int(ts), tz=timezone.utc)
                    if ts
                    else _to_utc(str(fx.get("date", "")))
                )
                teams = fx.get("teams") or {}
                home = teams.get("home") or {}
                is_home = int(home.get("id") or -1) == int(team_id)
                goals = fx.get("goals") or {}
                gf = int(goals.get("home") or 0) if is_home else int(goals.get("away") or 0)
                ga = int(goals.get("away") or 0) if is_home else int(goals.get("home") or 0)

                collected.append(
                    {
                        "fixture_id": int(f_id) if f_id is not None else None,
                        "date_utc": dt,
                        "home_away": "H" if is_home else "A",
                        "goals_for": gf,
                        "goals_against": ga,
                    }
                )
                needed -= 1
            cur_season -= 1

        return collected[:last]

    # --------------------------- League goal means (cheap) ---------------------------
    def league_goal_means(self, league_id: int, season: int) -> tuple[float, float]:
        """Return (mu_home, mu_away) as league-level goal averages for a season.

        Computes from finished fixtures and caches the result.
        """
        key = (int(league_id), int(season))
        cached = self._cache_league_means.get(key)
        if cached is not None:
            return cached
        fixtures = self.get_league_finished_fixtures(league_id, season)
        if not fixtures:
            # Sensible global prior if league data is unavailable
            prior = (1.35, 1.15)
            self._cache_league_means.set(key, prior)
            return prior
        h = [float(r.get("home_goals", 0.0)) for r in fixtures]
        a = [float(r.get("away_goals", 0.0)) for r in fixtures]
        mu_h = sum(h) / len(h) if h else 1.3
        mu_a = sum(a) / len(a) if a else 1.1
        out = (mu_h, mu_a)
        self._cache_league_means.set(key, out)
        return out

    # --------------------------- League finished fixtures for a season ---------------------------
    def get_league_finished_fixtures(self, league_id: int, season: int) -> list[dict[str, Any]]:
        """Return all finished (FT) fixtures for a league and season.

        Each item: {date_utc, home_id, away_id, home_goals, away_goals}
        """
        params = {"league": str(int(league_id)), "season": str(int(season)), "status": "FT"}
        first = self._client.get("fixtures", params)

        items: list[Mapping[str, Any]] = []
        if isinstance(first, Mapping):
            resp = first.get("response")
            if isinstance(resp, list):
                items.extend(resp)

        # Handle paging if present
        if isinstance(first, Mapping):
            paging = first.get("paging")
            total = 1
            if isinstance(paging, Mapping):
                raw_total = paging.get("total")
                if raw_total is not None:
                    try:
                        total = int(raw_total)
                    except (TypeError, ValueError):
                        total = 1
            for page in range(2, total + 1):
                more_params = dict(params)
                more_params["page"] = str(page)
                payload = self._client.get("fixtures", more_params)
                if isinstance(payload, Mapping):
                    resp = payload.get("response")
                    if isinstance(resp, list):
                        items.extend(resp)

        out: list[dict[str, Any]] = []
        for item in items:
            fx = item.get("fixture") or {}
            teams = item.get("teams") or {}
            goals = item.get("goals") or {}
            date_str = fx.get("date")
            dt = _to_utc(date_str) if isinstance(date_str, str) else datetime.now(timezone.utc)
            home = teams.get("home") or {}
            away = teams.get("away") or {}
            out.append(
                {
                    "date_utc": dt,
                    "home_id": _safe_int(home.get("id")),
                    "away_id": _safe_int(away.get("id")),
                    "home_goals": _safe_int(goals.get("home")) or 0,
                    "away_goals": _safe_int(goals.get("away")) or 0,
                }
            )

        # Sort by date for stable ELO traversal
        def _date_key(r: dict[str, Any]) -> datetime:
            v = r.get("date_utc")
            return v if isinstance(v, datetime) else datetime.now(timezone.utc)

        out.sort(key=_date_key)
        return out

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
                lg = item.get("league") or {}
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
                        "league": lg,
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
        cached = self._cache_fixture_stats.get(int(fixture_id))
        if cached is not None:
            return cached
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
        try:
            self._cache_fixture_stats.set(int(fixture_id), result)
        except Exception:
            pass
        return result

    # --------------------------- Fixture result ---------------------------
    def get_fixture_result_label(self, fixture_id: int) -> str | None:
        """Return '1'/'X'/'2' for a fixture when a final result is inferable.

        Accepts final statuses like 'FT' (full time), 'AET' (after extra time), 'PEN' (after penalties).
        Falls back to team winner flags or score breakdown if goals are missing.
        """
        try:
            payload = self._client.get("fixtures", {"id": str(int(fixture_id))})
        except Exception:
            return None
        if not isinstance(payload, Mapping):
            return None
        resp = payload.get("response")
        if not isinstance(resp, list) or not resp:
            return None
        item = resp[0]
        fx = item.get("fixture") or {}
        status = (
            (fx.get("status") or {}).get("short") if isinstance(fx.get("status"), Mapping) else None
        )
        teams = item.get("teams") or {}
        home_t = teams.get("home") or {}
        away_t = teams.get("away") or {}
        # If API already flags a winner team
        try:
            if home_t.get("winner") is True:
                return "1"
            if away_t.get("winner") is True:
                return "2"
        except Exception:
            pass

        # Accept a set of finished statuses
        finished_statuses = {"FT", "AET", "PEN"}
        if status not in finished_statuses:
            return None

        # Prefer top-level goals
        goals = item.get("goals") or {}
        gh = goals.get("home")
        ga = goals.get("away")
        try:
            gh_i = int(gh) if gh is not None else None
            ga_i = int(ga) if ga is not None else None
        except Exception:
            gh_i = ga_i = None

        # If goals are missing, try score breakdown
        if gh_i is None or ga_i is None:
            score = item.get("score") or {}
            for key in ("penalty", "extratime", "fulltime"):
                seg = score.get(key) or {}
                try:
                    _h_tmp = _safe_int(seg.get("home"))
                    _a_tmp = _safe_int(seg.get("away"))
                    if _h_tmp is not None:
                        gh_i = _h_tmp
                    if _a_tmp is not None:
                        ga_i = _a_tmp
                except Exception:
                    continue
                if gh_i is not None and ga_i is not None:
                    break

        if gh_i is None or ga_i is None:
            return None
        if gh_i == ga_i:
            return "X"
        return "1" if gh_i > ga_i else "2"

    # --------------------------- Team main league for a season ---------------------------
    def get_team_main_league(self, team_id: int, season: int) -> int | None:
        """Infer the league where a team mostly played in a given season.

        Queries fixtures by team+season (no league filter) and returns the most frequent league id.
        """
        params = {"team": str(int(team_id)), "season": str(int(season)), "status": "FT"}
        payload = self._client.get("fixtures", params)
        counts: dict[int, int] = {}
        if isinstance(payload, Mapping):
            resp = payload.get("response")
            if isinstance(resp, list):
                for item in resp:
                    lg = item.get("league") or {}
                    lid = _safe_int(lg.get("id"))
                    if lid is None:
                        continue
                    counts[int(lid)] = counts.get(int(lid), 0) + 1
        if not counts:
            return None
        # Return league id with max count
        return max(counts.items(), key=lambda kv: kv[1])[0]


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
