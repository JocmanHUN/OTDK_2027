"""Enum definitions for the interfaces layer.

This module defines enums used in the interfaces layer.
Since the project targets Python 3.11+, we can safely use
the built-in :class:`enum.StrEnum`.
"""

from enum import StrEnum


class PredictionStatus(StrEnum):
    """Status of a prediction.

    >>> PredictionStatus.OK.value
    'OK'
    """

    OK = "OK"
    SKIPPED = "SKIPPED"
