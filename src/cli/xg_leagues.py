from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Protocol, Sequence

from src.application.services.history_service import HistoryService
from src.application.services.leagues_service import LeagueSeason, LeaguesService

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = BASE_DIR / "data" / "leagues_with_xg.json"


def _fixture_has_xg(stats: Mapping[int, Mapping[str, float]]) -> bool:
    for team_stats in stats.values():
        for name, value in team_stats.items():
            if value is None:
                continue
            name_lc = name.lower()
            if "xg" in name_lc or ("expected" in name_lc and "goal" in name_lc):
                return True
    return False


def _iter_recent_fixture_ids(fixtures: Sequence[Mapping[str, object]]) -> Iterable[int]:
    def _ts(row: Mapping[str, object]) -> datetime:
        val = row.get("date_utc")
        if isinstance(val, datetime):
            return val
        return datetime.min.replace(tzinfo=timezone.utc)

    for row in sorted(fixtures, key=_ts, reverse=True):
        fx_id = row.get("fixture_id")
        if isinstance(fx_id, int):
            yield fx_id


class _HistoryProvider(Protocol):
    def get_league_finished_fixtures(
        self, league_id: int, season: int
    ) -> Sequence[Mapping[str, Any]]: ...

    def get_fixture_statistics(self, fixture_id: int) -> dict[int, dict[str, float]]: ...


@dataclass(frozen=True)
class _FilterConfig:
    min_checks: int = 3
    min_success: int = 3
    max_probe: int = 10
    stat_delay: float = 0.3

    def clamp(self) -> _FilterConfig:
        min_checks = max(1, self.min_checks)
        min_success = max(1, self.min_success)
        max_probe = max(min_checks, self.max_probe)
        min_checks = max(min_checks, min_success)
        stat_delay = max(0.0, float(self.stat_delay))
        return _FilterConfig(
            min_checks=min_checks,
            min_success=min_success,
            max_probe=max_probe,
            stat_delay=stat_delay,
        )


def find_leagues_with_xg(
    leagues: Sequence[LeagueSeason],
    history: _HistoryProvider,
    cfg: _FilterConfig,
) -> list[LeagueSeason]:
    """Return subset of leagues whose statistics expose non-null xG for enough fixtures."""

    if not leagues:
        return []

    cfg = cfg.clamp()
    qualified: list[LeagueSeason] = []

    for league in leagues:
        league_id = int(league["league_id"])
        season = int(league["season_year"])
        fixtures = history.get_league_finished_fixtures(league_id, season)
        if not fixtures:
            continue

        checked = 0
        successes = 0
        for fixture_id in _iter_recent_fixture_ids(fixtures):
            stats = history.get_fixture_statistics(fixture_id)
            if cfg.stat_delay > 0:
                time.sleep(cfg.stat_delay)
            checked += 1
            if _fixture_has_xg(stats):
                successes += 1
            if checked >= cfg.max_probe:
                break
            if checked >= cfg.min_checks and successes >= cfg.min_success:
                break

        if checked < cfg.min_checks:
            # Try to keep checking if we ran out before reaching min_checks
            continue
        if successes >= cfg.min_success:
            qualified.append(league)

    return qualified


def _write_output(leagues: Sequence[LeagueSeason], output: Path, cfg: _FilterConfig) -> None:
    payload: MutableMapping[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "min_fixture_checks": cfg.min_checks,
        "min_xg_hits": cfg.min_success,
        "max_probe": cfg.max_probe,
        "leagues": list(leagues),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect leagues whose fixture statistics include non-null xG values."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Target JSON file for qualified leagues (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--min-checks",
        type=int,
        default=3,
        help="Minimum FT fixtures to inspect per league (default: 3).",
    )
    parser.add_argument(
        "--min-success",
        type=int,
        default=3,
        help="Minimum fixtures that must expose xG to keep a league (default: 3).",
    )
    parser.add_argument(
        "--max-probe",
        type=int,
        default=10,
        help="Maximum fixtures to inspect per league before giving up (default: 10).",
    )
    parser.add_argument(
        "--stat-delay",
        type=float,
        default=0.3,
        help="Sleep (seconds) between consecutive fixture-stat requests to avoid rate limits (default: 0.3). "
        "Use 0 to disable.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = _FilterConfig(
        min_checks=args.min_checks,
        min_success=args.min_success,
        max_probe=args.max_probe,
        stat_delay=args.stat_delay,
    ).clamp()

    leagues = LeaguesService().get_current_leagues()
    history = HistoryService()

    qualified = find_leagues_with_xg(leagues, history, cfg)
    if not qualified:
        print("No leagues with reliable xG statistics were found.")
        return 1

    _write_output(qualified, Path(args.output), cfg)
    print(f"Wrote {len(qualified)} leagues with xG coverage to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
