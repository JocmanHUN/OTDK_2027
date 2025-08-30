from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

from src.application.services.history_service import HistoryService


def _fmt_dt(dt: object, tz: str) -> str:
    if isinstance(dt, datetime):
        try:
            return dt.astimezone(ZoneInfo(tz)).isoformat(timespec="minutes")
        except Exception:
            return dt.isoformat(timespec="minutes")
    return str(dt)


def _print_h2h(rows: Sequence[Mapping[str, object]], tz: str) -> None:
    if not rows:
        print("No head-to-head fixtures found.")
        return
    for r in rows:
        print(
            f"{_fmt_dt(r.get('date_utc'), tz)} | "
            f"{r.get('home_id')} vs {r.get('away_id')} -> "
            f"{r.get('home_goals')}-{r.get('away_goals')} ({r.get('result')})"
        )


def _format_all_stats(stats: Mapping[str, object]) -> list[str]:
    def pretty_key(k: str) -> str:
        return k.replace("_", " ")

    def format_value(key: str, v: object) -> str:
        # Ball possession és hasonló százalékos értékekre tegyünk % jelet, ha nincs
        perc_keys = {"ball possession", "passes %"}
        try:
            fv = float(v)  # type: ignore[arg-type]
            if key in perc_keys:
                return f"{fv:.0f}%"
            # integer, ha egész szám
            if abs(fv - int(fv)) < 1e-9:
                return f"{int(fv)}"
            return f"{fv:.2f}"
        except Exception:
            return str(v)

    lines: list[str] = []
    for k in sorted(stats.keys()):
        key = str(k)
        val = stats.get(k)
        lines.append(f"{pretty_key(key)}: {format_value(key, val)}")
    return lines


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="History utilities (H2H, team stats)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # h2h
    ph = sub.add_parser("h2h", help="List recent head-to-head fixtures")
    ph.add_argument("home", type=int, help="Home team ID")
    ph.add_argument("away", type=int, help="Away team ID")
    ph.add_argument("--last", type=int, default=20, help="How many recent fixtures")
    ph.add_argument(
        "--timezone",
        default="Europe/Budapest",
        help="Timezone to display dates (default: Europe/Budapest)",
    )

    # stats
    ps = sub.add_parser("stats", help="Show team statistics (averages)")
    ps.add_argument("--team", type=int, required=True, help="Team ID")
    ps.add_argument("--league", type=int, required=True, help="League ID")
    ps.add_argument("--season", type=int, required=True, help="Season year")
    ps.add_argument("--opponent", type=int, help="Optional opponent team ID for Poisson means")

    # recent
    pr = sub.add_parser(
        "recent", help="Last N league fixtures with per-match statistics (cross-season)"
    )
    pr.add_argument("--team", type=int, required=True, help="Team ID")
    pr.add_argument("--league", type=int, required=True, help="League ID")
    pr.add_argument("--season", type=int, required=True, help="Current season year")
    pr.add_argument("--last", type=int, required=True, help="How many recent fixtures to fetch")
    pr.add_argument(
        "--timezone",
        default="Europe/Budapest",
        help="Timezone to display dates (default: Europe/Budapest)",
    )
    pr.add_argument(
        "--only-finished",
        action="store_true",
        help="Include only finished (FT) fixtures when collecting recent matches",
    )
    pr.add_argument(
        "--json",
        action="store_true",
        help="Output full data as JSON (includes all per-match statistics)",
    )
    pr.add_argument(
        "--all-stats",
        action="store_true",
        help="Print all statistics for both teams in human-readable form",
    )

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    svc = HistoryService()

    if args.cmd == "h2h":
        rows = svc.get_head_to_head(args.home, args.away, last=args.last)
        _print_h2h(rows, tz=str(args.timezone))
        return 0

    if args.cmd == "stats":
        team_avg = svc.get_team_averages(args.team, args.league, args.season)
        print("Team averages:")
        print(
            f"matches_home={team_avg.matches_home} matches_away={team_avg.matches_away}\n"
            f"GF_home={team_avg.goals_for_home_avg:.2f} GF_away={team_avg.goals_for_away_avg:.2f}\n"
            f"GA_home={team_avg.goals_against_home_avg:.2f} GA_away={team_avg.goals_against_away_avg:.2f}"
        )
        if args.opponent is not None:
            opp = svc.get_team_averages(args.opponent, args.league, args.season)
            mu_h, mu_a = svc.simple_poisson_means(team_avg, opp)
            print(f"Poisson means vs opponent: mu_home={mu_h:.3f}, mu_away={mu_a:.3f}")
        return 0

    if args.cmd == "recent":
        rows = svc.get_recent_team_stats(
            args.team, args.league, args.season, args.last, only_finished=bool(args.only_finished)
        )
        if not rows:
            print("No fixtures found.")
            return 0
        if args.json:
            # Make rows JSON-serializable (datetime -> iso string)
            serializable = []
            for r in rows:
                rr = dict(r)
                dt = rr.get("date_utc")
                if isinstance(dt, datetime):
                    rr["date_utc"] = dt.isoformat()
                serializable.append(rr)
            print(json.dumps(serializable, ensure_ascii=False, indent=2))
            return 0

        tz = str(args.timezone)
        for r in rows:
            dt = _fmt_dt(r.get("date_utc"), tz)
            ha = r.get("home_away")
            opponent = r.get("opponent_id")
            gf = r.get("goals_for")
            ga = r.get("goals_against")
            match_stats = r.get("stats") or {}
            # Fejléc sor (összefoglaló)
            print(f"{dt} | {ha} vs {opponent} -> {gf}-{ga}")
            # Az adott csapat összes statjának szépen formázott listázása
            if isinstance(match_stats, dict) and match_stats:
                for line in _format_all_stats(match_stats):
                    print(f"  - {line}")
            if getattr(args, "all_stats", False):
                all_stats = r.get("all_stats") or {}
                # Print both teams' stats blocks
                if isinstance(all_stats, dict):
                    for tid, stats in all_stats.items():
                        print(f"  team={tid} stats:")
                        if isinstance(stats, dict):
                            # Sort keys for stability
                            for k in sorted(stats.keys()):
                                print(f"    - {k}: {stats.get(k)}")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
