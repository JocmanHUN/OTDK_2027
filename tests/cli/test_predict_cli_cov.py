# mypy: ignore-errors

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest


def _dummy_pred(
    match, status_value: str = "OK", probs: tuple[float, float, float] | None = (0.4, 0.3, 0.3)
):
    class P:
        def __init__(self) -> None:
            self.fixture_id = match.fixture_id
            self.model = type("M", (), {"value": "dummy"})
            self.version = "1"
            self.status = type("S", (), {"value": status_value})
            self.probs = (
                type("Pr", (), {"home": probs[0], "draw": probs[1], "away": probs[2]})()
                if probs is not None
                else None
            )
            self.skip_reason = "no data" if probs is None else None
            self.computed_at_utc = datetime.now(timezone.utc)

    return P()


def test_predict_daily_no_fixtures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.cli import predict_daily

    class DummySelector:
        def list_supported_leagues(self) -> list[dict]:
            return []

        def list_daily_matches(self, date: str, leagues: list[dict]) -> list[dict]:
            return []

        def filter_matches_with_1x2_odds(self, fixtures: list[dict]) -> list[dict]:
            return fixtures

    monkeypatch.setattr(
        predict_daily.SelectionPipeline, "default", staticmethod(lambda: DummySelector())
    )
    rc = predict_daily.main(["--date", "2024-01-01"])
    assert rc == 0
    assert "No fixtures to predict" in capsys.readouterr().out


def test_predict_daily_text_output_with_skip(
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
            # Return one OK with probs and one skipped
            return [_dummy_pred(match), _dummy_pred(match, status_value="SKIPPED", probs=None)]

    monkeypatch.setattr(
        predict_daily.SelectionPipeline, "default", staticmethod(lambda: DummySelector())
    )
    monkeypatch.setattr(predict_daily, "HistoryService", lambda: object())
    monkeypatch.setattr(predict_daily, "ContextBuilder", DummyCB)
    monkeypatch.setattr(predict_daily, "PredictionAggregatorImpl", lambda: DummyAgg())
    monkeypatch.setattr(predict_daily, "default_models", lambda: [object()])

    rc = predict_daily.main(["--date", "2024-01-01"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "model=dummy" in out
    assert "SKIPPED" in out


def test_predict_sample_no_fixtures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.cli import predict_sample

    class DummySelector:
        def list_supported_leagues(self) -> list[dict]:
            return [{"league_id": 1, "season_year": 2024}]

        def list_daily_matches(self, date: str, leagues: list[dict]) -> list[dict]:
            return []

        def filter_matches_with_1x2_odds(self, fixtures: list[dict]) -> list[dict]:
            return fixtures

    monkeypatch.setattr(
        predict_sample.SelectionPipeline, "default", staticmethod(lambda: DummySelector())
    )
    rc = predict_sample.main(["--date", "2024-01-01"])
    assert rc == 0
    assert "No fixtures to probe" in capsys.readouterr().out


def test_predict_sample_text_output(
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
            return [_dummy_pred(match)]

    monkeypatch.setattr(
        predict_sample.SelectionPipeline, "default", staticmethod(lambda: DummySelector())
    )
    monkeypatch.setattr(predict_sample, "HistoryService", lambda: object())
    monkeypatch.setattr(predict_sample, "ContextBuilder", DummyCB)
    monkeypatch.setattr(predict_sample, "PredictionAggregatorImpl", lambda: DummyAgg())
    monkeypatch.setattr(predict_sample, "default_models", lambda: [object()])

    rc = predict_sample.main(["--date", "2024-01-01", "--limit", "5"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "model=dummy" in out
