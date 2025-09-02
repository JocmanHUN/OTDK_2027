from __future__ import annotations

from datetime import datetime, timezone

from src.domain.entities.match import Match
from src.domain.interfaces.context import ModelContext
from src.domain.value_objects.enums import MatchStatus
from src.domain.value_objects.ids import FixtureId, LeagueId, TeamId
from src.models.veto import VetoModel


def _mk_match_ctx(
    *,
    league_id: int = 39,
    season: int = 2025,
    fixture_id: int = 100,
    home_id: int = 1,
    away_id: int = 2,
) -> tuple[Match, ModelContext]:
    match = Match(
        fixture_id=FixtureId(int(fixture_id)),
        league_id=LeagueId(int(league_id)),
        season=int(season),
        kickoff_utc=datetime.now(timezone.utc),
        home_name="H",
        away_name="A",
        status=MatchStatus.SCHEDULED,
    )
    ctx = ModelContext(
        fixture_id=FixtureId(int(fixture_id)),
        league_id=LeagueId(int(league_id)),
        season=int(season),
        home_team_id=TeamId(int(home_id)),
        away_team_id=TeamId(int(away_id)),
        home_goal_rate=None,
        away_goal_rate=None,
        elo_home=None,
        elo_away=None,
        home_advantage=0.0,
        features=None,
    )
    return match, ctx


def _rows_from_results(seq: list[str]) -> list[dict]:
    out: list[dict] = []
    for r in seq:
        if r == "W":
            gf, ga = 2, 0
        elif r == "D":
            gf, ga = 1, 1
        else:
            gf, ga = 0, 2
        out.append({"goals_for": gf, "goals_against": ga})
    return out


def _form_distribution(rows: list[dict], decay_factor: float) -> tuple[float, float, float]:
    if not rows:
        return (1 / 3, 1 / 3, 1 / 3)
    df = max(0.0, min(1.0, float(decay_factor)))
    if df == 0.0:
        df = 1.0
    wW = wD = wL = 0.0
    wsum = 0.0
    for i, r in enumerate(rows):
        w = df**i
        gf = int(r["goals_for"])
        ga = int(r["goals_against"])
        if gf > ga:
            wW += w
        elif gf == ga:
            wD += w
        else:
            wL += w
        wsum += w
    return (wW / wsum, wD / wsum, wL / wsum)


class _HistFake:
    def __init__(self, by_team: dict[int, list[dict]]):
        self._by_team = by_team

    def get_recent_team_stats(
        self, team_id: int, league_id: int, season: int, last: int, *, only_finished: bool = True
    ) -> list[dict]:  # noqa: E501
        seq = list(self._by_team.get(int(team_id), []))
        return seq[:last]


def test_veto_model_mixture_vs_average_and_product() -> None:
    home_seq = ["W", "D", "W", "L", "D"]
    away_seq = ["L", "D", "L", "W", "D"]
    df = 0.7
    last_n = 5
    w = 0.6
    hist = _HistFake(
        {
            10: _rows_from_results(home_seq),
            20: _rows_from_results(away_seq),
        }
    )
    model = VetoModel(history=hist, last_n=last_n, decay_factor=df, mul_weight=w)
    match, ctx = _mk_match_ctx(home_id=10, away_id=20)

    hW, hD, hL = _form_distribution(_rows_from_results(home_seq)[:last_n], df)
    aW, aD, aL = _form_distribution(_rows_from_results(away_seq)[:last_n], df)

    p1 = w * (hW * aL) + (1 - w) * 0.5 * (hW + aL)
    pX = w * (hD * aD) + (1 - w) * 0.5 * (hD + aD)
    p2 = w * (aW * hL) + (1 - w) * 0.5 * (aW + hL)
    norm = p1 + pX + p2
    exp_1, exp_X, exp_2 = p1 / norm, pX / norm, p2 / norm

    pred = model.predict(match, ctx)
    assert pred.probs is not None
    s = pred.probs.home + pred.probs.draw + pred.probs.away
    assert abs(s - 1.0) < 1e-12
    assert abs(pred.probs.home - exp_1) < 1e-12
    assert abs(pred.probs.draw - exp_X) < 1e-12
    assert abs(pred.probs.away - exp_2) < 1e-12


def test_veto_model_skips_without_history() -> None:
    match, ctx = _mk_match_ctx()
    model = VetoModel(history=None)
    pred = model.predict(match, ctx)
    assert pred.status.name == "SKIPPED"
