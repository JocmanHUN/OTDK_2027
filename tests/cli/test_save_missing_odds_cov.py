# mypy: ignore-errors

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from src.cli import save_missing_odds


def _seed_prediction(conn: sqlite3.Connection, match_id: int, *, league_id: int = 1) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO leagues(league_id, name, country) VALUES (?, 'L', 'C')", (league_id,)
    )
    conn.execute(
        "INSERT INTO matches(match_id, league_id, season, date, home_team, away_team) VALUES (?, ?, 2024, '2024-01-01', 'H', 'A')",
        (match_id, league_id),
    )
    conn.execute(
        "INSERT INTO predictions(match_id, model_name, prob_home, prob_draw, prob_away, predicted_result, is_correct, result_status) VALUES (?, 'm', 0.3, 0.3, 0.4, '1', NULL, 'PENDING')",
        (match_id,),
    )


def test_ensure_db_and_find_missing_odds(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    conn = save_missing_odds.ensure_db(str(db_path))

    _seed_prediction(conn, 101)
    _seed_prediction(conn, 102)
    conn.execute("INSERT INTO bookmakers(bookmaker_id, name) VALUES (5, 'B1')")
    conn.execute(
        "INSERT INTO odds(match_id, bookmaker_id, odds_home, odds_draw, odds_away) VALUES (102, 5, 1.1, 2.2, 3.3)"
    )

    missing = save_missing_odds.find_matches_missing_odds(conn)
    assert missing == [101]
    conn.close()


def test_persist_odds_handles_errors_and_inserts(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    conn = save_missing_odds.ensure_db(str(db_path))
    _seed_prediction(conn, 201)

    class _ErrorSvc:
        def get_fixture_bookmakers(self, fixture_id: int):
            raise RuntimeError("names boom")

        def get_fixture_odds(self, fixture_id: int):
            raise RuntimeError("odds boom")

    # Errors on both bookmaker and odds paths return 0
    assert save_missing_odds.persist_odds_for_match(conn, 201, _ErrorSvc()) == 0

    class _OkSvc:
        def get_fixture_bookmakers(self, fixture_id: int):
            return {"5": "Bookie"}

        def get_fixture_odds(self, fixture_id: int):
            return [
                SimpleNamespace(
                    fixture_id=fixture_id, bookmaker_id=5, home=1.1, draw=2.2, away=3.3
                ),
                SimpleNamespace(
                    fixture_id=fixture_id, bookmaker_id="bad", home="x", draw=2, away=3
                ),
            ]

    written = save_missing_odds.persist_odds_for_match(conn, 201, _OkSvc())
    assert written == 1
    cur = conn.execute(
        "SELECT match_id, odds_home, odds_draw, odds_away FROM odds WHERE match_id = 201"
    )
    rows = cur.fetchall()
    assert rows == [(201, 1.1, 2.2, 3.3)]
    conn.close()


def test_main_flow_inserts_missing_odds(monkeypatch, tmp_path, capsys) -> None:
    db_path = tmp_path / "app.db"
    conn = save_missing_odds.ensure_db(str(db_path))
    _seed_prediction(conn, 301)
    _seed_prediction(conn, 302)
    conn.close()

    class _Svc:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def get_fixture_bookmakers(self, fixture_id: int):
            return {"9": "Nine"}

        def get_fixture_odds(self, fixture_id: int):
            self.calls.append(int(fixture_id))
            return [
                SimpleNamespace(fixture_id=fixture_id, bookmaker_id=9, home=1.2, draw=3.4, away=5.6)
            ]

    monkeypatch.setattr(save_missing_odds, "OddsService", _Svc)

    rc = save_missing_odds.main(["--db", str(db_path), "--limit", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "saved 1 odds rows" in out

    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT COUNT(*) FROM odds")
    assert cur.fetchone()[0] == 1
    conn.close()


def test_main_no_missing_matches(monkeypatch, tmp_path, capsys) -> None:
    db_path = tmp_path / "app.db"
    conn = save_missing_odds.ensure_db(str(db_path))
    conn.close()

    class _Svc:
        def __init__(self) -> None:
            raise AssertionError("service should not be created")

    monkeypatch.setattr(save_missing_odds, "OddsService", _Svc)

    rc = save_missing_odds.main(["--db", str(db_path)])
    assert rc == 0
    assert "No matches missing odds found." in capsys.readouterr().out
