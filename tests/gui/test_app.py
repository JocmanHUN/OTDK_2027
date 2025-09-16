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


def test_predict_then_persist_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_db: Any) -> None:
    app = tmp_db
    conn = app._ensure_db()
    try:
        when = _tomorrow_utc_midday()
        fixture: dict[str, Any] = {
            "fixture_id": 202,
            "league_id": 2,
            "season": when.year,
            "date_utc": when,
            "home_name": "Home",
            "away_name": "Away",
            "home_id": 11,
            "away_id": 22,
        }

        # Patch ContextBuilder to avoid network
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
                    elo_home=1500.0,
                    elo_away=1500.0,
                )

        monkeypatch.setattr(app, "ContextBuilder", FakeCtxBuilder)

        # Two OK models
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

        monkeypatch.setattr(app, "default_models", fake_models)

        # Avoid network in odds persist: fake OddsService
        class FakeOddsSvc:
            def get_fixture_odds(self, fixture_id: int) -> list[Any]:
                return []

            def get_fixture_bookmakers(self, fixture_id: int) -> dict[int, str]:
                return {}

        monkeypatch.setattr(app, "OddsService", lambda: FakeOddsSvc())

        # Call twice, should persist once due to upserts and unique constraints
        assert app._predict_then_persist_if_complete(conn, fixture) is True
        assert app._predict_then_persist_if_complete(conn, fixture) is True

        cur = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE match_id = ?",
            (fixture["fixture_id"],),
        )
        assert cur.fetchone()[0] == 2

        cur2 = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE match_id = ?", (fixture["fixture_id"],)
        )
        assert cur2.fetchone()[0] == 1

        rows = conn.execute(
            "SELECT model_name, prob_home, prob_draw, prob_away FROM predictions WHERE match_id = ?",
            (fixture["fixture_id"],),
        ).fetchall()
        got = {r[0]: (r[1], r[2], r[3]) for r in rows}
        assert "poisson" in got and "elo" in got
        ph, pd, pa = got["poisson"]
        assert abs((ph + pd + pa) - 1.0) < 1e-9
        assert ph > pd > pa
    finally:
        conn.close()


def test_predict_then_persist_skipped_no_persist(
    monkeypatch: pytest.MonkeyPatch, tmp_db: Any
) -> None:
    app = tmp_db
    conn = app._ensure_db()
    try:
        when = _tomorrow_utc_midday()
        fixture: dict[str, Any] = {
            "fixture_id": 303,
            "league_id": 3,
            "season": when.year,
            "date_utc": when,
            "home_name": "H1",
            "away_name": "A1",
            "home_id": 13,
            "away_id": 23,
        }

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
                    home_goal_rate=1.0,
                    away_goal_rate=1.0,
                    elo_home=1500.0,
                    elo_away=1500.0,
                )

        monkeypatch.setattr(app, "ContextBuilder", FakeCtxBuilder)

        def fake_models() -> List[Any]:
            from dataclasses import dataclass
            from datetime import datetime, timezone

            from src.domain.entities.prediction import Prediction
            from src.domain.interfaces.modeling import BasePredictiveModel
            from src.domain.value_objects.enums import ModelName, PredictionStatus
            from src.domain.value_objects.probability_triplet import ProbabilityTriplet

            @dataclass
            class OKModel(BasePredictiveModel):
                name = ModelName.POISSON
                version = "ok"

                def predict(self, match: Any, ctx: Any) -> Any:
                    return Prediction(
                        fixture_id=match.fixture_id,
                        model=self.name,
                        probs=ProbabilityTriplet(home=0.6, draw=0.2, away=0.2),
                        computed_at_utc=datetime.now(timezone.utc),
                        version=self.version,
                        status=PredictionStatus.OK,
                    )

            @dataclass
            class SkippedModel(BasePredictiveModel):
                name = ModelName.ELO
                version = "skip"

                def predict(self, match: Any, ctx: Any) -> Any:
                    # Simulate skip: probs=None, SKIPPED
                    return Prediction(
                        fixture_id=match.fixture_id,
                        model=self.name,
                        probs=None,
                        computed_at_utc=datetime.now(timezone.utc),
                        version=self.version,
                        status=PredictionStatus.SKIPPED,
                        skip_reason="no data",
                    )

            return [OKModel(), SkippedModel()]

        monkeypatch.setattr(app, "default_models", fake_models)

        # Avoid network in odds persist: fake OddsService
        class FakeOddsSvc:
            def get_fixture_odds(self, fixture_id: int) -> list[Any]:
                return []

            def get_fixture_bookmakers(self, fixture_id: int) -> dict[int, str]:
                return {}

        monkeypatch.setattr(app, "OddsService", lambda: FakeOddsSvc())

        ok = app._predict_then_persist_if_complete(conn, fixture)
        assert ok is False

        # Should not persist any match or predictions
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM matches WHERE match_id = ?", (fixture["fixture_id"],)
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE match_id = ?", (fixture["fixture_id"],)
            ).fetchone()[0]
            == 0
        )
    finally:
        conn.close()
