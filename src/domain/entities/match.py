from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..value_objects.enums import MatchStatus
from ..value_objects.ids import FixtureId, LeagueId


class Match(BaseModel):
    fixture_id: FixtureId = Field(..., description="Unique identifier for the fixture")
    league_id: LeagueId = Field(..., description="League identifier")
    season: int = Field(..., ge=0, description="Season year")
    kickoff_utc: datetime = Field(..., description="Kickoff time in UTC")
    home_name: str = Field(..., description="Home team name")
    away_name: str = Field(..., description="Away team name")
    status: MatchStatus = Field(..., description="Current match status")
    ft_home_goals: int | None = Field(
        default=None, description="Full-time goals for home team if match finished"
    )
    ft_away_goals: int | None = Field(
        default=None, description="Full-time goals for away team if match finished"
    )

    model_config = ConfigDict(frozen=True)

    @field_validator("kickoff_utc", mode="before")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("kickoff_utc must be timezone-aware")
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _require_score_if_finished(self) -> "Match":
        if self.status == MatchStatus.FINISHED:
            if self.ft_home_goals is None or self.ft_away_goals is None:
                raise ValueError("Finished match must have full-time goals set")
        return self
