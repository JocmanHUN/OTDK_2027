from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Dict, Mapping, Optional, Tuple

from src.config.league_tiers import get_tier_config

from .history_service import HistoryService


@dataclass
class EloParams:
    k_prev: float = 16.0
    k_curr: float = 20.0
    regres_weight: float = 0.7  # weight on previous season end when seeding current season
    draw_param: float = 0.28  # fallback if league draw rate not estimated
    mov_scaling: bool = True
    promotion_offset: float = -30.0  # start below baseline when promoted into stronger tier
    relegation_offset: float = +30.0  # start above baseline when relegated into weaker tier


def _expected_score(delta_with_home: float) -> float:
    # logistic with base-10 elo scale 400
    d = -float(delta_with_home) / 400.0
    den = 1.0 + pow(10.0, d)
    return float(1.0 / den)


def _mov_factor(goal_diff: int, elo_diff_abs: float) -> float:
    # Bradley-Terry MOV scaling used in chess-like elo for football
    return 1.0 + log(goal_diff + 1.0) * (2.2 / (0.001 * elo_diff_abs + 2.2))


class EloService:
    """Compute ELO ratings for a league across seasons and cache results in-memory.

    Strategy:
    - Compute previous season ELO within the league from finished fixtures.
    - Seed current season by blending prev end and league tier baseline.
    - Update through current season finished fixtures.
    - Ratings are per (league_id, season_year) for all teams in that league season.

    Promotion/Relegation note:
    - Teams not present in previous season start from tier baseline.
    - Later we can adjust with +/- offset when detecting cross-league movement.
    """

    def __init__(
        self, history: Optional[HistoryService] = None, params: Optional[EloParams] = None
    ) -> None:
        self.history = history or HistoryService()
        self.params = params or EloParams()
        self._cache: Dict[Tuple[int, int], Dict[int, float]] = {}

    def get_team_rating(self, league_id: int, season: int, team_id: int) -> float:
        ratings = self.get_league_ratings(league_id, season)
        return ratings.get(int(team_id), float(get_tier_config(int(league_id)).base_elo))

    def get_league_ratings(self, league_id: int, season: int) -> Dict[int, float]:
        key = (int(league_id), int(season))
        if key in self._cache:
            return self._cache[key]

        prev_year = int(season) - 1
        # 1) Build prev season ratings
        prev = self._compute_season_ratings(league_id, prev_year, k=self.params.k_prev, seed=None)

        # 2) Seed current from regression to baseline
        cfg = get_tier_config(int(league_id))
        seeded: Dict[int, float] = {}
        if prev:
            for tid, rating in prev.items():
                seeded[tid] = (
                    self.params.regres_weight * rating
                    + (1.0 - self.params.regres_weight) * cfg.base_elo
                )

        # 2b) Handle teams new to the league (promotion/relegation heuristic)
        # Infer each team's main league in previous season to detect cross-league movement and adjust seed.
        # If a team didn't play this league in prev season:
        #   - If previous tier number > current tier number (moved to stronger league) -> promotion_offset
        #   - If previous tier number < current tier number (moved to weaker league) -> relegation_offset
        # Teams without prev season info remain at league baseline.
        cur_fixtures = self.history.get_league_finished_fixtures(int(league_id), int(season))
        cur_teams: Dict[int, int] = {}
        for row in cur_fixtures:
            hid = row.get("home_id")
            if isinstance(hid, int):
                cur_teams[int(hid)] = 1
            aid = row.get("away_id")
            if isinstance(aid, int):
                cur_teams[int(aid)] = 1
        for tid in list(cur_teams.keys()):
            if tid in seeded:
                continue
            prev_league = self.history.get_team_main_league(tid, int(prev_year))
            if prev_league is None:
                # Unknown -> baseline
                seeded[tid] = float(cfg.base_elo)
                continue
            prev_cfg = get_tier_config(int(prev_league))
            # Lower tier number means stronger league
            if prev_cfg.tier > cfg.tier:
                # Promoted to stronger league -> start a bit below baseline
                seeded[tid] = float(cfg.base_elo + self.params.promotion_offset)
            elif prev_cfg.tier < cfg.tier:
                # Relegated to weaker league -> start a bit above baseline
                seeded[tid] = float(cfg.base_elo + self.params.relegation_offset)
            else:
                seeded[tid] = float(cfg.base_elo)

        # 3) Compute current season, starting from seeded + baseline for new teams
        current = self._compute_season_ratings(
            league_id, int(season), k=self.params.k_curr, seed=seeded if seeded else None
        )
        self._cache[key] = current
        return current

    def _compute_season_ratings(
        self, league_id: int, season: int, *, k: float, seed: Optional[Mapping[int, float]]
    ) -> Dict[int, float]:
        fixtures = self.history.get_league_finished_fixtures(int(league_id), int(season))
        cfg = get_tier_config(int(league_id))
        ratings: Dict[int, float] = dict(seed) if seed else {}

        def get_r(team: Optional[int]) -> float:
            if team is None:
                return float(cfg.base_elo)
            return ratings.get(int(team), float(cfg.base_elo))

        for fx in fixtures:
            h = fx.get("home_id")
            a = fx.get("away_id")
            gh = int(fx.get("home_goals") or 0)
            ga = int(fx.get("away_goals") or 0)
            if h is None or a is None:
                continue
            rh = get_r(h)
            ra = get_r(a)

            delta = rh - ra
            # Apply home advantage as rating boost for home when computing expected
            e_home = _expected_score(delta + cfg.home_adv)
            # Real outcome
            if gh > ga:
                s_home = 1.0
            elif gh == ga:
                s_home = 0.5
            else:
                s_home = 0.0

            g = 1.0
            if self.params.mov_scaling and gh != ga:
                g = _mov_factor(abs(gh - ga), abs(delta))

            rh_new = rh + k * g * (s_home - e_home)
            ra_new = ra + k * g * ((1.0 - s_home) - (1.0 - e_home))
            ratings[int(h)] = rh_new
            ratings[int(a)] = ra_new

        return ratings
