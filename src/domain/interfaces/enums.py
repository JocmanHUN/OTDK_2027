"""Common enums used by interfaces.

This module re-exports :class:`PredictionStatus` from the value object layer so
all components reference the same enum instance.

>>> PredictionStatus.OK.value
'OK'
"""

from ..value_objects.enums import PredictionStatus

__all__ = ["PredictionStatus"]
