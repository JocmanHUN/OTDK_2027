# mypy: ignore-errors

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.repositories.bookmakers import Bookmaker
from src.repositories.leagues import League
from src.repositories.matches import Match
from src.repositories.odds import Odds
from src.repositories.predictions import Prediction
from src.repositories.sqlite.bookmakers_sqlite import BookmakersRepoSqlite
from src.repositories.sqlite.leagues_sqlite import LeaguesRepoSqlite
from src.repositories.sqlite.matches_sqlite import MatchesRepoSqlite
from src.repositories.sqlite.odds_sqlite import OddsRepoSqlite
from src.repositories.sqlite.predictions_sqlite import PredictionsRepoSqlite


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    sql = Path("migrations/V1__base.sql").read_text()
    conn.executescript(sql)
    return conn


def test_bookmakers_and_leagues_crud() -> None:
    conn = _conn()
    brepo = BookmakersRepoSqlite(conn)
    lrepo = LeaguesRepoSqlite(conn)

    bid1 = brepo.insert(Bookmaker(id=None, name="B1"))
    bid2 = brepo.insert(Bookmaker(id=99, name="B2"))
    assert brepo.get_by_id(bid1).name == "B1"
    brepo.update(Bookmaker(id=bid1, name="B1-upd"))
    assert brepo.get_by_id(bid1).name == "B1-upd"
    all_books = brepo.list_all(limit=10, offset=0)
    assert {b.id for b in all_books} == {bid1, 99}
    brepo.delete(bid2)
    assert brepo.get_by_id(bid2) is None

    lid = lrepo.insert(League(id=None, name="Premier", country="UK"))
    lrepo.insert(League(id=77, name="La Liga", country="ES"))
    assert lrepo.get_by_id(lid).name == "Premier"
    lrepo.update(League(id=lid, name="Premier League", country="GB"))
    assert lrepo.get_by_id(lid).country == "GB"
    leagues = lrepo.list_all(limit=5, offset=0)
    assert len(leagues) == 2
    lrepo.delete(77)
    assert lrepo.get_by_id(77) is None
    conn.close()


def test_bookmakers_leagues_rowid_none(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = sqlite3.connect(":memory:")
    b_repo = BookmakersRepoSqlite(conn)
    l_repo = LeaguesRepoSqlite(conn)

    class DummyCur:
        lastrowid = None

    class StubConn:
        def __init__(self, base: sqlite3.Connection) -> None:
            self.base = base

        def execute(self, *a, **kw):
            return DummyCur()

        def commit(self) -> None:
            pass

    # Swap connection to stub so insert raises via lastrowid None
    monkeypatch.setattr(b_repo, "_conn", StubConn(conn))
    with pytest.raises(RuntimeError):
        b_repo.insert(Bookmaker(id=None, name="X"))

    monkeypatch.setattr(l_repo, "_conn", StubConn(conn))
    with pytest.raises(RuntimeError):
        l_repo.insert(League(id=None, name="L", country="C"))


def test_matches_repo_flow() -> None:
    conn = _conn()
    # Seed league for FK
    conn.execute("INSERT INTO leagues(league_id, name, country) VALUES (1, 'L', 'C')")
    repo = MatchesRepoSqlite(conn)

    dt = datetime(2024, 1, 2, 3, 4, tzinfo=timezone.utc)
    mid = repo.insert(
        Match(id=None, league_id=1, season=2024, date=dt, home_team="H", away_team="A")
    )
    fetched = repo.get_by_id(mid)
    assert fetched is not None and fetched.home_team == "H"

    repo.update_result(mid, "1")
    assert repo.get_by_id(mid).real_result == "1"
    with pytest.raises(ValueError):
        repo.update_result(mid, "bad")  # type: ignore[arg-type]

    repo.insert(Match(id=None, league_id=1, season=2024, date=dt, home_team="X", away_team="Y"))
    listed = repo.list_by_league(1, 2024)
    assert len(listed) == 2
    repo.delete(mid)
    assert repo.get_by_id(mid) is None
    conn.close()


def test_matches_repo_handles_bad_date_strings() -> None:
    conn = _conn()
    conn.execute("INSERT INTO leagues(league_id, name, country) VALUES (2, 'L', 'C')")
    conn.execute(
        "INSERT INTO matches(match_id, league_id, season, date, home_team, away_team) VALUES (50, 2, 2024, 'not-a-date', 'H', 'A')"
    )
    repo = MatchesRepoSqlite(conn)
    row = repo.get_by_id(50)
    assert row is not None
    assert row.date == datetime.fromtimestamp(0)
    listed = repo.list_by_league(2, 2024)
    assert listed and listed[0].date == datetime.fromtimestamp(0)
    conn.close()


def test_matches_repo_datetime_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _conn()
    conn.execute("INSERT INTO leagues(league_id, name, country) VALUES (4, 'L', 'C')")
    repo = MatchesRepoSqlite(conn)

    dt = datetime(2024, 7, 1, tzinfo=timezone.utc)

    class Cur:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

        def fetchall(self):
            return [self._row]

    class StubConn:
        def __init__(self, row):
            self.row = row

        def execute(self, *a, **kw):
            return Cur(self.row)

        def commit(self) -> None:
            pass

    row = (10, 4, 2024, dt, "H", "A", None)
    repo._conn = StubConn(row)  # type: ignore[assignment]
    res = repo.get_by_id(10)
    assert res is not None and res.date == dt
    rows = repo.list_by_league(4, 2024)
    assert rows and rows[0].date == dt


def test_matches_rowid_none(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _conn()
    conn.execute("INSERT INTO leagues(league_id, name, country) VALUES (3, 'L', 'C')")
    repo = MatchesRepoSqlite(conn)

    class DummyCur:
        lastrowid = None

    class StubConn:
        def execute(self, *a, **kw):
            return DummyCur()

        def commit(self) -> None:
            pass

    repo._conn = StubConn()  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        repo.insert(
            Match(
                id=None,
                league_id=3,
                season=2024,
                date=datetime.now(timezone.utc),
                home_team="H",
                away_team="A",
            )
        )


def test_odds_repo_upsert_and_best() -> None:
    conn = _conn()
    conn.execute("INSERT INTO leagues(league_id, name, country) VALUES (1, 'L', 'C')")
    conn.execute(
        "INSERT INTO matches(match_id, league_id, season, date, home_team, away_team) VALUES (10, 1, 2024, '2024-01-01', 'H', 'A')"
    )
    conn.execute("INSERT INTO bookmakers(bookmaker_id, name) VALUES (5, 'B1'), (6, 'B2')")

    repo = OddsRepoSqlite(conn)
    oid = repo.insert(Odds(id=None, match_id=10, bookmaker_id=5, home=1.5, draw=3.0, away=4.0))
    # Upsert should update existing row for same match/bookmaker
    repo.insert(Odds(id=None, match_id=10, bookmaker_id=5, home=1.8, draw=2.9, away=3.9))
    stored = repo.get_by_id(oid)
    assert stored is not None and stored.home == 1.8

    repo.insert(Odds(id=None, match_id=10, bookmaker_id=6, home=1.6, draw=3.5, away=4.5))
    best = repo.best_odds(10)
    # Home best is bookmaker 5 with 1.8, draw/away from bookmaker 6
    assert best["1"] == (5, 1.8)
    assert best["X"][0] in {5, 6}
    assert best["2"][0] in {5, 6}
    listed = repo.list_by_match(10)
    assert len(listed) == 2
    # Update path
    repo.update(Odds(id=listed[0].id, match_id=10, bookmaker_id=5, home=2.0, draw=3.1, away=4.2))
    assert repo.get_by_id(listed[0].id).home == 2.0
    repo.delete(oid)
    assert repo.get_by_id(oid) is None
    conn.close()


def test_odds_repo_dedup_on_init() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(
        """
        CREATE TABLE odds (
            odds_id INTEGER PRIMARY KEY,
            match_id INTEGER NOT NULL,
            bookmaker_id INTEGER NOT NULL,
            odds_home REAL NOT NULL,
            odds_draw REAL NOT NULL,
            odds_away REAL NOT NULL
        )
        """
    )
    # Duplicate (match_id, bookmaker_id) rows to trigger IntegrityError on unique index creation
    conn.execute(
        "INSERT INTO odds(odds_id, match_id, bookmaker_id, odds_home, odds_draw, odds_away) VALUES (1, 1, 1, 1.1, 2.2, 3.3)"
    )
    conn.execute(
        "INSERT INTO odds(odds_id, match_id, bookmaker_id, odds_home, odds_draw, odds_away) VALUES (2, 1, 1, 1.2, 2.1, 3.0)"
    )
    OddsRepoSqlite(conn)
    rows = conn.execute("SELECT odds_id, odds_home FROM odds").fetchall()
    # Only latest row remains after dedup
    assert rows == [(2, 1.2)]
    # Second init should not raise once index exists
    OddsRepoSqlite(conn)
    conn.close()


def test_odds_rowid_none_and_integrity(monkeypatch: pytest.MonkeyPatch) -> None:
    base_conn = sqlite3.connect(":memory:")
    base_conn.executescript(Path("migrations/V1__base.sql").read_text())

    class ProxyConn:
        def __init__(self, base: sqlite3.Connection) -> None:
            self.base = base
            self.raise_index = True

        def execute(self, sql: str, params=()):
            if "CREATE UNIQUE INDEX" in sql and self.raise_index:
                self.raise_index = False
                raise sqlite3.IntegrityError("dup")
            return self.base.execute(sql, params)

        def commit(self) -> None:
            self.base.commit()

    proxied = ProxyConn(base_conn)
    repo = OddsRepoSqlite(proxied)

    class DummyCur:
        lastrowid = None

    monkeypatch.setattr(
        repo,
        "_conn",
        type(
            "Stub", (), {"execute": lambda *a, **kw: DummyCur(), "commit": lambda self=None: None}
        )(),
    )
    with pytest.raises(RuntimeError):
        repo.insert(Odds(id=None, match_id=1, bookmaker_id=1, home=1.1, draw=2.2, away=3.3))


def test_predictions_repo_upsert_and_mark_correct() -> None:
    conn = _conn()
    conn.execute("INSERT INTO leagues(league_id, name, country) VALUES (1, 'L', 'C')")
    conn.execute(
        "INSERT INTO matches(match_id, league_id, season, date, home_team, away_team) VALUES (20, 1, 2024, '2024-02-02', 'H', 'A')"
    )
    repo = PredictionsRepoSqlite(conn)

    pid = repo.insert(
        Prediction(
            id=None,
            match_id=20,
            model_name="M1",
            prob_home=0.6,
            prob_draw=0.2,
            prob_away=0.2,
            predicted_result="1",
        )
    )
    first = repo.get_by_id(pid)
    assert first is not None and abs(first.prob_home + first.prob_draw + first.prob_away - 1) < 1e-9

    # Upsert with different probs and result_status
    repo.insert(
        Prediction(
            id=None,
            match_id=20,
            model_name="M1",
            prob_home=0.3,
            prob_draw=0.3,
            prob_away=0.4,
            predicted_result="2",
            is_correct=None,
            result_status="PENDING",
        )
    )
    listed = repo.list_by_match(20)
    assert len(listed) == 1 and listed[0].predicted_result == "2"

    repo.mark_correct(listed[0].id, True)
    updated = repo.get_by_id(listed[0].id)
    assert updated is not None and updated.result_status == "WIN" and updated.is_correct is True

    with pytest.raises(ValueError):
        repo.update(
            Prediction(
                id=listed[0].id,
                match_id=20,
                model_name="M1",
                prob_home=0.0,
                prob_draw=0.0,
                prob_away=0.0,
                predicted_result="1",
            )
        )

    repo.delete(listed[0].id)
    assert repo.get_by_id(listed[0].id) is None
    conn.close()


def test_predictions_repo_legacy_schema_dedup_and_alter() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    # Legacy table without result_status and with duplicate rows for unique index failure
    conn.execute(
        """
        CREATE TABLE predictions (
            prediction_id INTEGER PRIMARY KEY,
            match_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            prob_home REAL NOT NULL,
            prob_draw REAL NOT NULL,
            prob_away REAL NOT NULL,
            predicted_result TEXT NOT NULL,
            is_correct INTEGER
        )
        """
    )
    conn.execute(
        "INSERT INTO predictions(prediction_id, match_id, model_name, prob_home, prob_draw, prob_away, predicted_result, is_correct) VALUES (1, 1, 'M', 0.5, 0.3, 0.2, '1', 1)"
    )
    conn.execute(
        "INSERT INTO predictions(prediction_id, match_id, model_name, prob_home, prob_draw, prob_away, predicted_result, is_correct) VALUES (2, 1, 'M', 0.6, 0.2, 0.2, '2', 0)"
    )
    PredictionsRepoSqlite(conn)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    assert "result_status" in cols
    # Duplicate should be deduped to keep latest prediction_id=2
    rows = conn.execute("SELECT prediction_id, match_id, model_name FROM predictions").fetchall()
    assert rows == [(2, 1, "M")]
    # Unique index exists (no IntegrityError on creating another repo)
    PredictionsRepoSqlite(conn)
    conn.close()


def test_predictions_repo_invalid_probs_raise() -> None:
    conn = _conn()
    repo = PredictionsRepoSqlite(conn)
    with pytest.raises(ValueError):
        repo.insert(
            Prediction(
                id=None,
                match_id=1,
                model_name="M",
                prob_home=0.0,
                prob_draw=0.0,
                prob_away=0.0,
                predicted_result="1",
            )
        )
    conn.close()


def test_predictions_repo_update_upserts() -> None:
    conn = _conn()
    conn.execute("INSERT INTO leagues(league_id, name, country) VALUES (1, 'L', 'C')")
    conn.execute(
        "INSERT INTO matches(match_id, league_id, season, date, home_team, away_team) VALUES (5, 1, 2024, '2024-03-03', 'H', 'A')"
    )
    repo = PredictionsRepoSqlite(conn)
    pid = repo.insert(
        Prediction(
            id=None,
            match_id=5,
            model_name="M5",
            prob_home=0.7,
            prob_draw=0.2,
            prob_away=0.1,
            predicted_result="1",
        )
    )
    # Update same (match, model) with different probs and status
    repo.update(
        Prediction(
            id=pid,
            match_id=5,
            model_name="M5",
            prob_home=0.5,
            prob_draw=0.25,
            prob_away=0.25,
            predicted_result="X",
            is_correct=True,
            result_status="WIN",
        )
    )
    row = repo.get_by_id(pid)
    assert row is not None
    assert row.predicted_result == "X"
    assert row.result_status == "WIN"
    conn.close()


def test_predictions_rowid_none_integrity_and_db_error(monkeypatch: pytest.MonkeyPatch) -> None:
    base_conn = sqlite3.connect(":memory:")
    base_conn.executescript(Path("migrations/V1__base.sql").read_text())

    class ProxyConn:
        def __init__(self, base: sqlite3.Connection) -> None:
            self.base = base
            self.raise_index = True

        def execute(self, sql: str, params=()):
            if "table_info" in sql:
                raise sqlite3.DatabaseError("bad pragma")
            if "CREATE UNIQUE INDEX" in sql:
                if self.raise_index:
                    self.raise_index = False
                    raise sqlite3.IntegrityError("dup index")
            return self.base.execute(sql, params)

        def commit(self) -> None:
            self.base.commit()

    proxy = ProxyConn(base_conn)
    repo = PredictionsRepoSqlite(proxy)

    class DummyCur:
        lastrowid = None

    monkeypatch.setattr(
        repo,
        "_conn",
        type(
            "Stub", (), {"execute": lambda *a, **kw: DummyCur(), "commit": lambda self=None: None}
        )(),
    )
    with pytest.raises(RuntimeError):
        repo.insert(
            Prediction(
                id=None,
                match_id=1,
                model_name="X",
                prob_home=0.5,
                prob_draw=0.3,
                prob_away=0.2,
                predicted_result="1",
            )
        )


def test_predictions_rounding_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _conn()
    conn.execute("INSERT INTO leagues(league_id, name, country) VALUES (9, 'L', 'C')")
    conn.execute(
        "INSERT INTO matches(match_id, league_id, season, date, home_team, away_team) VALUES (9, 9, 2024, '2024-04-04', 'H', 'A')"
    )
    repo = PredictionsRepoSqlite(conn)

    # Monkeypatch round to return a large value so ih + id_ exceeds SCALE
    monkeypatch.setattr("builtins.round", lambda x: (1 << 20))

    pid = repo.insert(
        Prediction(
            id=None,
            match_id=9,
            model_name="M9",
            prob_home=0.500001,
            prob_draw=0.500001,
            prob_away=0.1,
            predicted_result="1",
        )
    )
    repo.update(
        Prediction(
            id=pid,
            match_id=9,
            model_name="M9",
            prob_home=0.500001,
            prob_draw=0.500001,
            prob_away=0.1,
            predicted_result="1",
        )
    )
    row = repo.get_by_id(pid)
    assert row is not None
    assert row.prob_draw in (0.0, 0)  # branch adjusted draw component to zero
