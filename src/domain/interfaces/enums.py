"""Enum definitions for the interfaces layer.

- Python 3.11+: use enum.StrEnum
- Python 3.10 : provide a small local fallback (str + Enum)
"""

import sys
from enum import Enum

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        """Backport of enum.StrEnum for Python < 3.11."""

        pass


class PredictionStatus(StrEnum):
    """Status of a prediction."""

    OK = "OK"
    SKIPPED = "SKIPPED"
