"""Application settings for API-FOOTBALL access.

This module centralises configuration for reaching the API-FOOTBALL service.
Environment variables are loaded from a ``.env`` file using ``python-dotenv``
and exposed through a Pydantic settings object.
"""

from __future__ import annotations

import os
from typing import Mapping

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

# Load environment variables from a .env file if present
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIRECT_BASE_URL = "https://v3.football.api-sports.io/"
DIRECT_HOST = "v3.football.api-sports.io"
RAPIDAPI_BASE_URL = "https://api-football-v1.p.rapidapi.com/v3/"
REQUEST_TIMEOUT = 10  # seconds


class Settings(BaseModel):
    """Immutable settings object used across the application."""

    base_url: str
    headers: Mapping[str, str]
    timeout: int = REQUEST_TIMEOUT

    model_config = ConfigDict(frozen=True)


def _build_settings() -> Settings:
    """Construct the ``Settings`` instance based on environment variables."""

    use_rapidapi = os.getenv("USE_RAPIDAPI", "false").lower() == "true"

    api_key = os.getenv("API_KEY")
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    rapidapi_host = os.getenv("RAPIDAPI_HOST", "api-football-v1.p.rapidapi.com")

    if use_rapidapi:
        if not rapidapi_key:
            raise RuntimeError("RAPIDAPI_KEY is required when USE_RAPIDAPI=true")
        base_url = RAPIDAPI_BASE_URL
        headers: Mapping[str, str] = {
            "x-rapidapi-host": rapidapi_host,
            "x-rapidapi-key": rapidapi_key,
        }
    else:
        if not api_key:
            if rapidapi_key is None:
                raise RuntimeError("No API access key provided for API-FOOTBALL")
            raise RuntimeError(
                "API_KEY is required for direct API-FOOTBALL access. Set USE_RAPIDAPI=true to"
                " use RapidAPI fallback."
            )
        base_url = DIRECT_BASE_URL
        headers = {"x-apisports-key": api_key}

    return Settings(base_url=base_url, headers=headers)


# Public settings instance
settings = _build_settings()

# Backwards compatibility / convenient exports
BASE_URL = settings.base_url
HOST = DIRECT_HOST
