from __future__ import annotations

from typing import Any

import pytest


def test_poisson_cli_smoke(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.cli import poisson

    class DummyCtx:
        pass

    class DummyCB:
        def __init__(self, history: Any) -> None:
            self.history = history

        def build_from_meta(self, **_: Any) -> DummyCtx:
            return DummyCtx()

    class DummyAgg:
        def run_all(self, models: list[Any], match: Any, ctx: Any) -> list[Any]:
            class P:
                def __init__(self) -> None:
                    self.fixture_id = match.fixture_id
                    self.model = "poisson"
                    self.version = "1"
                    self.status = type("S", (), {"value": "OK"})  # simple enum-like
                    self.probs = type("Pr", (), {"home": 0.5, "draw": 0.3, "away": 0.2})()
                    self.skip_reason = None

            return [P()]

    monkeypatch.setattr(poisson, "HistoryService", lambda: object())
    monkeypatch.setattr(poisson, "ContextBuilder", DummyCB)
    monkeypatch.setattr(poisson, "PredictionAggregatorImpl", lambda: DummyAgg())
    monkeypatch.setattr(poisson, "PoissonModel", lambda: object())

    code = poisson.main(
        ["--fixture", "1", "--league", "2", "--season", "2024", "--home", "3", "--away", "4"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Poisson probabilities" in out


def test_predict_daily_cli_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.cli import predict_daily

    class DummySelector:
        def list_supported_leagues(self) -> list[dict]:
            return [{"league_id": 1, "season_year": 2024}]

        def list_daily_matches(self, date: str, leagues: list[dict]) -> list[dict]:
            return [
                {
                    "fixture_id": 10,
                    "league_id": leagues[0]["league_id"],
                    "season_year": leagues[0]["season_year"],
                    "home_id": 5,
                    "away_id": 6,
                }
            ]

        def filter_matches_with_1x2_odds(self, fixtures: list[dict]) -> list[dict]:
            return fixtures

    class DummyCtx:
        pass

    class DummyCB:
        def __init__(self, history: Any) -> None:
            self.history = history

        def build_from_meta(self, **_: Any) -> DummyCtx:
            return DummyCtx()

    class DummyAgg:
        def run_all(self, models: list[Any], match: Any, ctx: Any) -> list[Any]:
            class P:
                def __init__(self) -> None:
                    self.fixture_id = match.fixture_id
                    self.model = type("M", (), {"value": "dummy"})
                    self.version = "1"
                    self.status = type("S", (), {"value": "OK"})
                    self.probs = type("Pr", (), {"home": 0.4, "draw": 0.3, "away": 0.3})()
                    self.skip_reason = None
                    from datetime import datetime, timezone

                    self.computed_at_utc = datetime.now(timezone.utc)

            return [P()]

    monkeypatch.setattr(
        predict_daily.SelectionPipeline, "default", staticmethod(lambda: DummySelector())
    )
    monkeypatch.setattr(predict_daily, "HistoryService", lambda: object())
    monkeypatch.setattr(predict_daily, "ContextBuilder", DummyCB)
    monkeypatch.setattr(predict_daily, "PredictionAggregatorImpl", lambda: DummyAgg())
    monkeypatch.setattr(predict_daily, "default_models", lambda: [object()])

    code = predict_daily.main(["--date", "2024-01-01", "--json"])
    out = capsys.readouterr().out.strip()
    assert code == 0
    assert out.startswith("[") and '"fixture_id": 10' in out


def test_predict_sample_cli_limit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.cli import predict_sample

    class DummySelector:
        def list_supported_leagues(self) -> list[dict]:
            return [{"league_id": 1, "season_year": 2024}]

        def list_daily_matches(self, date: str, leagues: list[dict]) -> list[dict]:
            return [
                {
                    "fixture_id": 10,
                    "league_id": leagues[0]["league_id"],
                    "season_year": leagues[0]["season_year"],
                    "home_id": 5,
                    "away_id": 6,
                }
            ]

        def filter_matches_with_1x2_odds(self, fixtures: list[dict]) -> list[dict]:
            return fixtures

    class DummyCtx:
        pass

    class DummyCB:
        def __init__(self, history: Any) -> None:
            self.history = history

        def build_from_meta(self, **_: Any) -> DummyCtx:
            return DummyCtx()

    class DummyAgg:
        def run_all(self, models: list[Any], match: Any, ctx: Any) -> list[Any]:
            class P:
                def __init__(self) -> None:
                    self.fixture_id = match.fixture_id
                    self.model = type("M", (), {"value": "dummy"})
                    self.version = "1"
                    self.status = type("S", (), {"value": "OK"})
                    self.probs = type("Pr", (), {"home": 0.5, "draw": 0.25, "away": 0.25})()
                    self.skip_reason = None
                    from datetime import datetime, timezone

                    self.computed_at_utc = datetime.now(timezone.utc)

            return [P()]

    monkeypatch.setattr(
        predict_sample.SelectionPipeline, "default", staticmethod(lambda: DummySelector())
    )
    monkeypatch.setattr(predict_sample, "HistoryService", lambda: object())
    monkeypatch.setattr(predict_sample, "ContextBuilder", DummyCB)
    monkeypatch.setattr(predict_sample, "PredictionAggregatorImpl", lambda: DummyAgg())
    monkeypatch.setattr(predict_sample, "default_models", lambda: [object()])

    code = predict_sample.main(["--date", "2024-01-01", "--limit", "1", "--json"])
    out = capsys.readouterr().out.strip()
    assert code == 0
    assert out.startswith("[") and '"fixture_id": 10' in out


def test_save_missing_odds_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.cli import save_missing_odds

    class DummyConn:
        def close(self) -> None:
            pass

    monkeypatch.setattr(save_missing_odds, "ensure_db", lambda db: DummyConn())
    monkeypatch.setattr(save_missing_odds, "find_matches_missing_odds", lambda conn: [1, 2, 3])
    monkeypatch.setattr(save_missing_odds, "persist_odds_for_match", lambda conn, fx, svc: 2)
    monkeypatch.setattr(save_missing_odds, "OddsService", lambda: object())

    code = save_missing_odds.main(["--db", "dummy.db", "--limit", "2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "saved 2 odds rows" in out
    assert "Matches: 2" in out
