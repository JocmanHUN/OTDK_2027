from __future__ import annotations

from datetime import datetime, timezone

from src.domain.entities.match import Match
from src.domain.interfaces.context import ModelContext
from src.domain.value_objects.enums import MatchStatus
from src.domain.value_objects.ids import FixtureId, LeagueId, TeamId
from src.models.elo import EloModel


def _mk_match_ctx(
    *,
    elo_home: float | None,
    elo_away: float | None,
) -> tuple[Match, ModelContext]:
    match = Match(
        fixture_id=FixtureId(3),
        league_id=LeagueId(39),
        season=2025,
        kickoff_utc=datetime.now(timezone.utc),
        home_name="H",
        away_name="A",
        status=MatchStatus.SCHEDULED,
    )
    ctx = ModelContext(
        fixture_id=FixtureId(3),
        league_id=LeagueId(39),
        season=2025,
        home_team_id=None,
        away_team_id=None,
        elo_home=elo_home,
        elo_away=elo_away,
    )
    return match, ctx


def test_elo_skips_without_team_ids_and_no_elos() -> None:
    match, ctx = _mk_match_ctx(elo_home=None, elo_away=None)
    model = EloModel()
    pred = model.predict(match, ctx)
    assert pred.status.name == "SKIPPED"


def test_elo_uses_provided_elos_and_clamps_draw_param() -> None:
    match, ctx = _mk_match_ctx(elo_home=1600.0, elo_away=1600.0)

    class _FakeSvc:
        def get_team_rating(self, league_id: int, season: int, team_id: int) -> float:
            return 1500.0

    # Negative draw_param should be clamped internally; still valid normalized result
    model = EloModel(draw_param=-1.0, elo_service=_FakeSvc())
    pred = model.predict(match, ctx)
    assert pred.probs is not None
    total = pred.probs.home + pred.probs.draw + pred.probs.away
    assert abs(total - 1.0) < 1e-12


def test_elo_uniform_when_league_ratings_missing() -> None:
    match = Match(
        fixture_id=FixtureId(9),
        league_id=LeagueId(5000),
        season=2025,
        kickoff_utc=datetime.now(timezone.utc),
        home_name="H",
        away_name="A",
        status=MatchStatus.SCHEDULED,
    )
    ctx = ModelContext(
        fixture_id=FixtureId(9),
        league_id=LeagueId(5000),
        season=2025,
        home_team_id=TeamId(1),
        away_team_id=TeamId(2),
    )

    class _EmptySvc:
        def get_team_rating(self, league_id: int, season: int, team_id: int) -> float:
            return 1500.0

        def get_league_ratings(self, league_id: int, season: int) -> dict[int, float]:
            return {}

    model = EloModel(elo_service=_EmptySvc())
    pred = model.predict(match, ctx)
    assert pred.probs is not None
    assert abs(pred.probs.home - 1.0 / 3.0) < 1e-12
    assert abs(pred.probs.draw - 1.0 / 3.0) < 1e-12
    assert abs(pred.probs.away - 1.0 / 3.0) < 1e-12
