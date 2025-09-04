from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List

import pytest


@pytest.fixture()
def tmp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    # Import here to access module-level constants for patching
    import src.gui.app as app

    db_path = tmp_path / "app.db"
    monkeypatch.setattr(app, "DB_PATH", str(db_path))
    # Reuse real migration file
    yield app


def test_ensure_db_creates_schema(tmp_db: Any) -> None:
    app = tmp_db
    conn = app._ensure_db()
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?,?,?,?)",
            ("matches", "predictions", "leagues", "odds"),
        )
        names = {r[0] for r in cur.fetchall()}
        assert {"matches", "predictions", "leagues", "odds"}.issubset(names)
    finally:
        conn.close()


def _tomorrow_utc_midday() -> datetime:
    today = datetime.now(timezone.utc).date()
    return datetime.combine(
        today + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
    ) + timedelta(hours=12)


def test_insert_matches_idempotent(tmp_db: Any) -> None:
    app = tmp_db
    conn = app._ensure_db()
    try:
        when = _tomorrow_utc_midday()
        fixtures: list[dict[str, Any]] = [
            {
                "fixture_id": 101,
                "league_id": 1,
                "season": when.year,
                "date_utc": when,
                "home_name": "Alpha",
                "away_name": "Beta",
                "home_id": 10,
                "away_id": 20,
            },
            {
                # duplicate fixture_id -> should not duplicate row
                "fixture_id": 101,
                "league_id": 1,
                "season": when.year,
                "date_utc": when,
                "home_name": "Alpha",
                "away_name": "Beta",
                "home_id": 10,
                "away_id": 20,
            },
        ]

        app._insert_matches(conn, fixtures)
        app._insert_matches(conn, fixtures)

        cur = conn.execute("SELECT COUNT(*) FROM matches WHERE match_id=101")
        assert cur.fetchone()[0] == 1
    finally:
        conn.close()


def test_predict_and_store_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_db: Any) -> None:
    app = tmp_db
    conn = app._ensure_db()
    try:
        # Insert one match for tomorrow
        when = _tomorrow_utc_midday()
        fixtures: list[dict[str, Any]] = [
            {
                "fixture_id": 202,
                "league_id": 2,
                "season": when.year,
                "date_utc": when,
                "home_name": "Home",
                "away_name": "Away",
                "home_id": 11,
                "away_id": 22,
            }
        ]
        app._insert_matches(conn, fixtures)

        # Patch ContextBuilder and default_models to avoid network and to be deterministic
        class FakeCtxBuilder:
            def __init__(self, history: Any | None = None) -> None:
                pass

            def build_from_meta(self, **kwargs: Any) -> Any:
                from src.domain.interfaces.context import ModelContext
                from src.domain.value_objects.ids import FixtureId, LeagueId, TeamId

                return ModelContext(
                    fixture_id=FixtureId(int(kwargs["fixture_id"])),
                    league_id=LeagueId(int(kwargs["league_id"])),
                    season=int(kwargs["season"]),
                    home_team_id=TeamId(int(kwargs["home_team_id"])),
                    away_team_id=TeamId(int(kwargs["away_team_id"])),
                    home_goal_rate=1.2,
                    away_goal_rate=0.8,
                )

        monkeypatch.setattr(app, "ContextBuilder", FakeCtxBuilder)

        # Two simple fake models with stable outputs
        def fake_models() -> List[Any]:
            from dataclasses import dataclass
            from datetime import datetime, timezone

            from src.domain.entities.prediction import Prediction
            from src.domain.interfaces.modeling import BasePredictiveModel
            from src.domain.value_objects.enums import ModelName, PredictionStatus
            from src.domain.value_objects.probability_triplet import ProbabilityTriplet

            @dataclass
            class M1(BasePredictiveModel):
                name = ModelName.POISSON
                version = "t1"

                def predict(self, match: Any, ctx: Any) -> Any:
                    return Prediction(
                        fixture_id=match.fixture_id,
                        model=self.name,
                        probs=ProbabilityTriplet(home=0.5, draw=0.3, away=0.2),
                        computed_at_utc=datetime.now(timezone.utc),
                        version=self.version,
                        status=PredictionStatus.OK,
                    )

            @dataclass
            class M2(BasePredictiveModel):
                name = ModelName.ELO
                version = "t2"

                def predict(self, match: Any, ctx: Any) -> Any:
                    return Prediction(
                        fixture_id=match.fixture_id,
                        model=self.name,
                        probs=ProbabilityTriplet(home=0.4, draw=0.3, away=0.3),
                        computed_at_utc=datetime.now(timezone.utc),
                        version=self.version,
                        status=PredictionStatus.OK,
                    )

            return [M1(), M2()]

        # Patch the models factory used inside the GUI module
        monkeypatch.setattr(app, "default_models", fake_models)

        # Execute twice to check idempotency (unique constraint + upsert)
        app._predict_and_store(conn, fixtures)
        app._predict_and_store(conn, fixtures)

        cur = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE match_id = ?",
            (fixtures[0]["fixture_id"],),
        )
        # Exactly two models stored, no duplicates
        assert cur.fetchone()[0] == 2

        # Verify probabilities are as expected and sum to 1 exactly per schema
        rows = conn.execute(
            "SELECT model_name, prob_home, prob_draw, prob_away FROM predictions WHERE match_id = ?",
            (fixtures[0]["fixture_id"],),
        ).fetchall()
        got = {r[0]: (r[1], r[2], r[3]) for r in rows}
        assert "poisson" in got and "elo" in got
        ph, pd, pa = got["poisson"]
        assert abs((ph + pd + pa) - 1.0) < 1e-9
        assert ph > pd > pa
    finally:
        conn.close()
