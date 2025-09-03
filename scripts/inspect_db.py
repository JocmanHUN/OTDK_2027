from __future__ import annotations

import argparse
import os
import sqlite3


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def list_dates(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute("SELECT DISTINCT substr(date,1,10) AS d FROM matches ORDER BY d")
    return [r["d"] for r in cur.fetchall()]


def list_matches_for_date(
    conn: sqlite3.Connection, day: str, *, only_with_odds: bool = True
) -> list[sqlite3.Row]:
    if only_with_odds:
        sql = (
            "SELECT m.match_id, m.league_id, m.season, m.date, m.home_team, m.away_team "
            "FROM matches m WHERE substr(m.date,1,10) = ? AND EXISTS(SELECT 1 FROM odds o WHERE o.match_id = m.match_id) "
            "ORDER BY m.date"
        )
    else:
        sql = (
            "SELECT match_id, league_id, season, date, home_team, away_team "
            "FROM matches WHERE substr(date,1,10) = ? ORDER BY date"
        )
    return list(conn.execute(sql, (day,)).fetchall())


def count_odds(conn: sqlite3.Connection, match_id: int) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM odds WHERE match_id = ?", (match_id,))
    return int(cur.fetchone()[0])


def count_predictions(conn: sqlite3.Connection, match_id: int) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM predictions WHERE match_id = ?", (match_id,))
    return int(cur.fetchone()[0])


def show_match_details(conn: sqlite3.Connection, match_id: int) -> None:
    m = conn.execute(
        "SELECT match_id, league_id, season, date, home_team, away_team FROM matches WHERE match_id = ?",
        (match_id,),
    ).fetchone()
    if not m:
        print(f"No match with id={match_id} found.")
        return
    print(
        f"Match {m['match_id']} | {m['date']} | L{m['league_id']} {m['season']} | {m['home_team']} - {m['away_team']}"
    )

    print("\nBookmaker odds:")
    for r in conn.execute(
        """
        SELECT o.bookmaker_id, COALESCE(b.name,'') AS name, o.odds_home, o.odds_draw, o.odds_away
        FROM odds o
        LEFT JOIN bookmakers b ON b.bookmaker_id = o.bookmaker_id
        WHERE o.match_id = ?
        ORDER BY o.bookmaker_id
        """,
        (match_id,),
    ):
        print(
            f"  {r['bookmaker_id']:>4} {r['name']:<25} 1:{r['odds_home']:.3f} X:{r['odds_draw']:.3f} 2:{r['odds_away']:.3f}"
        )

    print("\nPredictions:")
    for r in conn.execute(
        """
        SELECT model_name, prob_home, prob_draw, prob_away, predicted_result, result_status
        FROM predictions WHERE match_id = ? ORDER BY model_name
        """,
        (match_id,),
    ):
        ph, pd, pa = r["prob_home"], r["prob_draw"], r["prob_away"]
        status = r["result_status"] if r["result_status"] else "PENDING"
        print(
            f"  {r['model_name']:<24} home:{ph:.3f} draw:{pd:.3f} away:{pa:.3f} -> {r['predicted_result']} [{status}]"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Inspect SQLite DB contents")
    p.add_argument("--db", default=os.path.join("data", "odtk.sqlite3"))
    p.add_argument("--date", help="YYYY-MM-DD to list matches and counts")
    p.add_argument(
        "--include-no-odds",
        action="store_true",
        help="Include matches without stored odds in date listing",
    )
    p.add_argument("--match-id", type=int, help="Show detailed odds and predictions for a match")
    args = p.parse_args(argv)

    conn = _connect(args.db)
    try:
        if args.match_id is not None:
            show_match_details(conn, int(args.match_id))
            return 0

        if not args.date:
            ds = list_dates(conn)
            if not ds:
                print("No matches stored yet.")
                return 0
            print("Stored dates:")
            for d in ds:
                print(f"  {d}")
            print("\nProvide --date YYYY-MM-DD to list matches.")
            return 0

        rows = list_matches_for_date(conn, args.date, only_with_odds=not args.include_no_odds)
        if not rows:
            print(f"No matches for date {args.date}.")
            return 0
        print(f"Matches on {args.date}:")
        for r in rows:
            oidc = count_odds(conn, int(r["match_id"]))
            pc = count_predictions(conn, int(r["match_id"]))
            print(
                f"  {r['match_id']}: {r['home_team']} - {r['away_team']}  [{r['date']}]  odds:{oidc} preds:{pc}"
            )
        print("\nUse --match-id <id> for detailed view.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
