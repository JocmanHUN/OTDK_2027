"""Enum definitions for the interfaces layer.

This module previously relied on :class:`enum.StrEnum`, which is only
available starting from Python 3.11.  The project needs to run on earlier
Python versions as well, so we provide a small compatibility layer that
defines :class:`StrEnum` when it's not available.  The fallback behaves like a
regular ``Enum`` whose members are also ``str`` instances.
"""

try:  # pragma: no cover - import error handling is trivial
    from enum import StrEnum
except ImportError:  # Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of :class:`enum.StrEnum` for older Python versions."""


class PredictionStatus(StrEnum):
    """Status of a prediction.

    >>> PredictionStatus.OK.value
    'OK'
    """

    OK = "OK"
    SKIPPED = "SKIPPED"
