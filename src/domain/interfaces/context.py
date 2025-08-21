from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..value_objects.enums import ModelName
from ..value_objects.ids import FixtureId, LeagueId, TeamId


class ModelContext(BaseModel):
    """Input data shared across predictive models.

    >>> ctx = ModelContext(fixture_id=FixtureId(1), league_id=LeagueId(1), season=2024)
    >>> ctx.has_minimal_inputs_for(ModelName.POISSON)
    False
    """

    fixture_id: FixtureId
    league_id: LeagueId
    season: int
    home_team_id: TeamId | None = None
    away_team_id: TeamId | None = None
    home_goal_rate: float | None = None
    away_goal_rate: float | None = None
    elo_home: float | None = None
    elo_away: float | None = None
    home_advantage: float | None = None
    features: dict[str, float] | None = None

    model_config = ConfigDict(frozen=True)

    def has_minimal_inputs_for(self, model: ModelName) -> bool:
        """Return whether minimal inputs exist for ``model``.

        The check is heuristic; models may enforce stricter rules.

        >>> ctx = ModelContext(
        ...     fixture_id=FixtureId(1),
        ...     league_id=LeagueId(1),
        ...     season=2024,
        ...     home_goal_rate=1.2,
        ...     away_goal_rate=0.8,
        ... )
        >>> ctx.has_minimal_inputs_for(ModelName.POISSON)
        True
        """

        match model:
            case ModelName.POISSON:
                return self.home_goal_rate is not None and self.away_goal_rate is not None
            case ModelName.ELO:
                return self.elo_home is not None and self.elo_away is not None
            case ModelName.LOGISTIC_REGRESSION:
                return self.features is not None
            case _:
                return True
