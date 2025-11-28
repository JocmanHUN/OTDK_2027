from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

BASE_EXPORT_ROOT = Path(__file__).resolve().parents[3] / "data" / "training_exports"


def _daily_file(kind: str, date_str: str, base_dir: Path | None) -> Path:
    root = Path(base_dir) if base_dir is not None else BASE_EXPORT_ROOT
    folder = root / kind
    folder.mkdir(parents=True, exist_ok=True)
    safe_date = str(date_str).replace("/", "-")
    return folder / f"{safe_date}.csv"


def _append_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _iso_dt(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    try:
        return datetime.fromisoformat(str(value)).astimezone(timezone.utc).isoformat()
    except Exception:
        return ""


def log_odds_csv(
    run_date: str,
    fixture_meta: Mapping[str, Any],
    odds_list: Sequence[Any],
    *,
    base_dir: Path | None = None,
) -> None:
    """Append odds rows for a fixture to the daily odds CSV."""
    if not odds_list:
        return

    path = _daily_file("odds", run_date, base_dir)
    fieldnames = [
        "run_date",
        "fixture_id",
        "league_id",
        "season",
        "bookmaker_id",
        "odds_home",
        "odds_draw",
        "odds_away",
        "collected_at_utc",
    ]
    ts = datetime.now(timezone.utc).isoformat()
    fixture_id = _safe_int(fixture_meta.get("fixture_id"))
    league_id = _safe_int(fixture_meta.get("league_id"))
    season = _safe_int(fixture_meta.get("season") or fixture_meta.get("season_year"))

    rows: list[dict[str, Any]] = []
    for o in odds_list:
        bid = _safe_int(getattr(o, "bookmaker_id", None))
        h = _safe_float(getattr(o, "home", None))
        d = _safe_float(getattr(o, "draw", None))
        a = _safe_float(getattr(o, "away", None))
        if bid is None or h is None or d is None or a is None:
            continue
        rows.append(
            {
                "run_date": run_date,
                "fixture_id": fixture_id,
                "league_id": league_id,
                "season": season,
                "bookmaker_id": bid,
                "odds_home": h,
                "odds_draw": d,
                "odds_away": a,
                "collected_at_utc": ts,
            }
        )
    _append_csv(path, fieldnames, rows)


def log_features_csv(
    run_date: str,
    fixture_meta: Mapping[str, Any],
    ctx: Any,
    *,
    base_dir: Path | None = None,
) -> None:
    """Append aggregated features for a fixture to the daily features CSV."""
    path = _daily_file("features", run_date, base_dir)
    fieldnames = [
        "run_date",
        "fixture_id",
        "league_id",
        "season",
        "date_utc",
        "home_id",
        "away_id",
        "home_name",
        "away_name",
        "home_goal_rate",
        "away_goal_rate",
        "features_json",
        "collected_at_utc",
    ]

    fixture_id = _safe_int(fixture_meta.get("fixture_id"))
    league_id = _safe_int(fixture_meta.get("league_id"))
    season = _safe_int(fixture_meta.get("season") or fixture_meta.get("season_year"))
    dt_raw = fixture_meta.get("date_utc")
    home_id = _safe_int(fixture_meta.get("home_id"))
    away_id = _safe_int(fixture_meta.get("away_id"))

    feats = getattr(ctx, "features", None)
    feats_json = json.dumps(feats or {}, ensure_ascii=False)
    row = {
        "run_date": run_date,
        "fixture_id": fixture_id,
        "league_id": league_id,
        "season": season,
        "date_utc": _iso_dt(dt_raw),
        "home_id": home_id,
        "away_id": away_id,
        "home_name": fixture_meta.get("home_name"),
        "away_name": fixture_meta.get("away_name"),
        "home_goal_rate": _safe_float(getattr(ctx, "home_goal_rate", None)),
        "away_goal_rate": _safe_float(getattr(ctx, "away_goal_rate", None)),
        "features_json": feats_json,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _append_csv(path, fieldnames, [row])


def log_result_csv(
    run_date: str,
    match_meta: Mapping[str, Any],
    result_label: str,
    *,
    base_dir: Path | None = None,
) -> None:
    """Append a finished match result to the daily results CSV."""
    if result_label not in {"1", "X", "2"}:
        return
    path = _daily_file("results", run_date, base_dir)
    fieldnames = [
        "run_date",
        "fixture_id",
        "league_id",
        "season",
        "date_utc",
        "home_team",
        "away_team",
        "result",
        "logged_at_utc",
    ]

    row = {
        "run_date": run_date,
        "fixture_id": _safe_int(match_meta.get("fixture_id") or match_meta.get("id")),
        "league_id": _safe_int(match_meta.get("league_id")),
        "season": _safe_int(match_meta.get("season")),
        "date_utc": _iso_dt(match_meta.get("date_utc") or match_meta.get("date")),
        "home_team": match_meta.get("home_team") or match_meta.get("home_name"),
        "away_team": match_meta.get("away_team") or match_meta.get("away_name"),
        "result": result_label,
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _append_csv(path, fieldnames, [row])
