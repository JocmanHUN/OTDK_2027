from __future__ import annotations

import argparse
import os
import sqlite3
from collections import Counter, defaultdict
from typing import Iterable


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _iter_evaluable(
    conn: sqlite3.Connection, *, date_prefix: str | None = None, model: str | None = None
) -> Iterable[sqlite3.Row]:
    sql = (
        "SELECT p.prediction_id, p.match_id, p.model_name, p.predicted_result, p.result_status, "
        "       m.real_result, m.date "
        "FROM predictions p "
        "JOIN matches m ON m.match_id = p.match_id "
        "WHERE m.real_result IN ('1','X','2')"
    )
    params: list[object] = []
    if date_prefix:
        sql += " AND substr(m.date,1,10) = ?"
        params.append(date_prefix)
    if model:
        sql += " AND p.model_name = ?"
        params.append(model)
    sql += " ORDER BY m.date, p.match_id, p.model_name"
    cur = conn.execute(sql, params)
    for row in cur.fetchall():
        yield row


def _update_one(conn: sqlite3.Connection, prediction_id: int, is_correct: bool) -> None:
    status = "WIN" if is_correct else "LOSE"
    conn.execute(
        "UPDATE predictions SET is_correct = ?, result_status = ? WHERE prediction_id = ?",
        (int(is_correct), status, int(prediction_id)),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Evaluate predictions against match results and update status"
    )
    p.add_argument("--db", default=os.path.join("data", "odtk.sqlite3"))
    p.add_argument("--date", help="YYYY-MM-DD to restrict evaluation to a specific day")
    p.add_argument("--model", help="Only evaluate a single model name (exact match)")
    p.add_argument(
        "--dry-run", action="store_true", help="Do not write changes, only print summary"
    )
    args = p.parse_args(argv)

    conn = _connect(args.db)
    try:
        rows = list(_iter_evaluable(conn, date_prefix=args.date, model=args.model))
        if not rows:
            print("No predictions to evaluate for given filters.")
            return 0

        by_model: dict[str, Counter] = defaultdict(Counter)
        total_updates = 0

        if args.dry_run:
            for r in rows:
                correct = str(r["predicted_result"]) == str(
                    r["real_result"]
                )  # both present per query
                by_model[str(r["model_name"])]["WIN" if correct else "LOSE"] += 1
            # Print summary only
        else:
            with conn:
                for r in rows:
                    correct = str(r["predicted_result"]) == str(
                        r["real_result"]
                    )  # both present per query
                    _update_one(conn, int(r["prediction_id"]), correct)
                    by_model[str(r["model_name"])]["WIN" if correct else "LOSE"] += 1
                    total_updates += 1

        print(
            "Evaluation summary"
            + (f" for {args.date}" if args.date else "")
            + (f" model={args.model}" if args.model else "")
        )
        grand: Counter[str] = Counter()
        for model_name in sorted(by_model.keys()):
            c = by_model[model_name]
            wins = int(c.get("WIN", 0))
            loses = int(c.get("LOSE", 0))
            total = wins + loses
            acc = (wins / total) if total else 0.0
            grand.update(c)
            print(f"- {model_name}: total={total} win={wins} lose={loses} acc={acc:.3f}")
        g_wins = int(grand.get("WIN", 0))
        g_loses = int(grand.get("LOSE", 0))
        g_total = g_wins + g_loses
        g_acc = (g_wins / g_total) if g_total else 0.0
        print(f"Overall: total={g_total} win={g_wins} lose={g_loses} acc={g_acc:.3f}")
        if not args.dry_run:
            print(f"Updated rows: {total_updates}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
