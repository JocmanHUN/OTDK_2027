from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TierConfig:
    tier: int
    base_elo: int
    home_adv: int


# Seeded league tiers (API-FOOTBALL league IDs)
# This is an initial heuristic mapping and can be refined easily.
# Includes some strong second tiers (e.g., ENG Championship, 2. Bundesliga).

TIER_CONFIGS: Dict[int, TierConfig] = {
    # Tier 1 (Top 5) – Big 5
    39: TierConfig(1, 1600, 80),  # England Premier League
    140: TierConfig(1, 1600, 80),  # Spain La Liga
    135: TierConfig(1, 1600, 80),  # Italy Serie A
    78: TierConfig(1, 1600, 80),  # Germany Bundesliga
    61: TierConfig(1, 1600, 80),  # France Ligue 1
    # Tier 2 (6–10)
    88: TierConfig(2, 1550, 75),  # Netherlands Eredivisie
    94: TierConfig(2, 1550, 75),  # Portugal Primeira Liga
    144: TierConfig(2, 1550, 75),  # Belgium Pro League
    203: TierConfig(2, 1550, 75),  # Turkey Super Lig
    179: TierConfig(2, 1550, 75),  # Scotland Premiership
    # Strong second tiers promoted above many first tiers
    40: TierConfig(2, 1550, 75),  # England Championship
    79: TierConfig(2, 1550, 75),  # Germany 2. Bundesliga
    # Tier 3 (11–20)
    62: TierConfig(3, 1500, 70),  # France Ligue 2
    136: TierConfig(3, 1500, 70),  # Italy Serie B
    141: TierConfig(3, 1500, 70),  # Spain Segunda Division
    90: TierConfig(3, 1500, 70),  # Netherlands Eerste Divisie
    71: TierConfig(3, 1500, 70),  # Denmark Superliga
    1406: TierConfig(3, 1500, 70),  # Switzerland Super League (new IDs vary)
    197: TierConfig(3, 1500, 70),  # Austria Bundesliga
    384: TierConfig(3, 1500, 70),  # Czech First League
    566: TierConfig(3, 1500, 70),  # Croatia HNL
    501: TierConfig(3, 1500, 70),  # Greece Super League
    # Tier 4 (others default)
    # Add more as needed; unspecified leagues will fall back to Tier 4 baseline.
}


DEFAULT_BASE_ELO = 1450
DEFAULT_HOME_ADV = 65


def get_tier_config(league_id: int) -> TierConfig:
    cfg = TIER_CONFIGS.get(int(league_id))
    if cfg is not None:
        return cfg
    # Fallback to Tier 4 baseline
    return TierConfig(4, DEFAULT_BASE_ELO, DEFAULT_HOME_ADV)
