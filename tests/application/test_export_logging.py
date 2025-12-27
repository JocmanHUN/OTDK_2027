# mypy: ignore-errors

from __future__ import annotations

import csv
from datetime import datetime, timezone
from types import SimpleNamespace

from src.application.services import export_logging


def test_helpers_and_append_csv(tmp_path) -> None:
    path = tmp_path / "odds.csv"

    # Empty input keeps file untouched
    export_logging._append_csv(path, ["a"], [])
    assert not path.exists()

    export_logging._append_csv(
        path,
        ["one", "two"],
        [{"one": 1, "two": 2}, {"one": 3, "two": 4, "extra": "ignored"}],
    )
    rows = list(csv.DictReader(path.open()))
    assert rows == [{"one": "1", "two": "2"}, {"one": "3", "two": "4"}]

    assert export_logging._safe_int("bad") is None
    assert export_logging._safe_float("bad") is None
    assert export_logging._iso_dt("not-a-date") == ""

    dt = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert export_logging._iso_dt(dt).startswith("2024-01-02T03:04:05")


def test_log_odds_and_results_csv(tmp_path) -> None:
    class _Odd(SimpleNamespace):
        pass

    odds = [
        _Odd(bookmaker_id=None, home=1.2, draw=3.4, away=5.6),  # skipped
        _Odd(bookmaker_id="7", home="1.1", draw="2.2", away="3.3"),
    ]
    fixture_meta = {"fixture_id": "10", "league_id": "20", "season_year": "2024"}
    export_logging.log_odds_csv("2025-01-02", fixture_meta, odds, base_dir=tmp_path)

    odds_path = tmp_path / "odds" / "2025-01-02.csv"
    odds_rows = list(csv.DictReader(odds_path.open()))
    assert len(odds_rows) == 1
    assert odds_rows[0]["bookmaker_id"] == "7"
    assert odds_rows[0]["odds_home"] == "1.1"

    match_meta = {
        "id": "44",
        "league_id": "55",
        "season": "2023",
        "date": "2023-03-04T11:00:00+00:00",
        "home_name": "Home",
        "away_name": "Away",
    }
    # Ignore invalid label
    export_logging.log_result_csv("2025-01-02", match_meta, "?", base_dir=tmp_path)
    assert not (tmp_path / "results" / "2025-01-02.csv").exists()

    export_logging.log_result_csv("2025-01-02", match_meta, "X", base_dir=tmp_path)
    res_path = tmp_path / "results" / "2025-01-02.csv"
    res_rows = list(csv.DictReader(res_path.open()))
    assert res_rows[0]["result"] == "X"
    assert res_rows[0]["fixture_id"] == "44"
