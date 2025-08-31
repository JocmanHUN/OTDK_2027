from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Mapping


@dataclass
class FeaturesService:
    history: Any

    def build_features(
        self,
        home_team_id: int,
        away_team_id: int,
        league_id: int,
        season: int,
        *,
        last: int = 10,
    ) -> dict[str, float]:
        """Aggregate recent per-team statistics and return feature diffs (home - away).

        - Uses last N finished league fixtures for each team (cross-season within HistoryService).
        - Averages each numeric stat over the team samples.
        - Produces features as differences: diff_{stat} = avg_home(stat) - avg_away(stat).
        - Adds convenience aggregates: goals_for_avg, goals_against_avg, points_per_game.
        """
        home_rows = self.history.get_recent_team_stats(
            home_team_id, league_id, season, last, only_finished=True
        )
        away_rows = self.history.get_recent_team_stats(
            away_team_id, league_id, season, last, only_finished=True
        )

        def avg_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
            buckets: dict[str, list[float]] = {}
            gf: list[float] = []
            ga: list[float] = []
            pts: list[float] = []
            for r in rows:
                gf_val = float(r.get("goals_for") or 0)
                ga_val = float(r.get("goals_against") or 0)
                gf.append(gf_val)
                ga.append(ga_val)
                if gf_val > ga_val:
                    pts.append(3.0)
                elif gf_val == ga_val:
                    pts.append(1.0)
                else:
                    pts.append(0.0)
                stats = r.get("stats") or {}
                if isinstance(stats, Mapping):
                    for k, v in stats.items():
                        try:
                            fv = float(v)
                        except (TypeError, ValueError):
                            continue
                        buckets.setdefault(str(k), []).append(fv)
            out: dict[str, float] = {}
            if gf:
                out["goals_for_avg"] = mean(gf)
            if ga:
                out["goals_against_avg"] = mean(ga)
            if pts:
                out["points_per_game"] = mean(pts)
            for k, arr in buckets.items():
                if arr:
                    out[k] = float(mean(arr))
            return out

        avg_home = avg_stats(home_rows)
        avg_away = avg_stats(away_rows)

        # Build diff features for union of keys
        keys = set(avg_home.keys()) | set(avg_away.keys())
        features: dict[str, float] = {}
        for k in keys:
            hv = float(avg_home.get(k, 0.0))
            av = float(avg_away.get(k, 0.0))
            features[f"diff_{k}"] = hv - av
        return features
