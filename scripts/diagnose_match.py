from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _ensure_import_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _get_fixture_meta(fixture_id: int) -> dict[str, Any]:
    # Use the low-level client to fetch a single fixture by id
    from src.infrastructure.api_football_client import APIFootballClient

    client = APIFootballClient()
    payload = client.get("fixtures", {"id": str(int(fixture_id))})
    if isinstance(payload, Mapping):
        resp = payload.get("response")
        if isinstance(resp, list) and resp:
            item = resp[0]
            fx = item.get("fixture") or {}
            lg = item.get("league") or {}
            teams = item.get("teams") or {}
            home = teams.get("home") or {}
            away = teams.get("away") or {}
            date_str = fx.get("date")
            dt = _to_utc(date_str) if isinstance(date_str, str) else datetime.now(timezone.utc)
            return {
                "fixture_id": int(fixture_id),
                "date_utc": dt,
                "league_id": _safe_int(lg.get("id")) or 0,
                "season": _safe_int(lg.get("season")) or 0,
                "home_id": _safe_int(home.get("id")),
                "home_name": home.get("name"),
                "away_id": _safe_int(away.get("id")),
                "away_name": away.get("name"),
            }
    raise SystemExit(f"Fixture {fixture_id} not found or API returned unexpected payload")


def main(argv: list[str] | None = None) -> int:
    _ensure_import_path()

    p = argparse.ArgumentParser(description="Diagnose inputs for a fixture (lambdas, recent, h2h)")
    p.add_argument("--fixture-id", type=int, required=True)
    args = p.parse_args(argv)

    meta = _get_fixture_meta(int(args.fixture_id))

    from src.application.services.history_service import HistoryService
    from src.application.services.prediction_pipeline import ContextBuilder

    fx_id = int(meta["fixture_id"])  # noqa: F841
    league_id = int(meta["league_id"])
    season = int(meta["season"])
    home_id_raw = meta.get("home_id")
    away_id_raw = meta.get("away_id")
    if home_id_raw is None or away_id_raw is None:
        print("Missing team IDs in fixture metadata; cannot diagnose.")
        return 1
    home_id = int(home_id_raw)
    away_id = int(away_id_raw)

    print(
        f"Fixture {fx_id} | L{league_id} {season} | "
        f"{meta.get('home_name')} ({home_id}) vs {meta.get('away_name')} ({away_id}) | {meta.get('date_utc')}"
    )

    history = HistoryService()

    # Recent league matches (finished, friendlies excluded) counts
    recent_home = history.get_recent_team_stats(int(home_id), league_id, season, 10)
    recent_away = history.get_recent_team_stats(int(away_id), league_id, season, 10)
    print(
        f"Recent (league, excl. friendlies): home={len(recent_home)} away={len(recent_away)} (target 10)"
    )

    # H2H count (friendlies excluded)
    h2h_rows = history.get_head_to_head(
        int(home_id), int(away_id), last=50, exclude_friendlies=True
    )
    print(f"H2H (excl. friendlies): count={len(h2h_rows)} (target >=5)")

    # Lambdas via ContextBuilder (uses HistoryService.simple_poisson_means)
    ctx = ContextBuilder(history=history).build_from_meta(
        fixture_id=int(args.fixture_id),
        league_id=league_id,
        season=season,
        home_team_id=int(home_id),
        away_team_id=int(away_id),
    )
    print(
        f"Lambdas (Poisson inputs): home_goal_rate={ctx.home_goal_rate:.4f} "
        f"away_goal_rate={ctx.away_goal_rate:.4f}"
    )

    return 0


def _safe_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_utc(iso_str: str) -> datetime:
    s = iso_str.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.fromisoformat(s.split(".")[0]) if "." in s else datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


if __name__ == "__main__":
    raise SystemExit(main())
