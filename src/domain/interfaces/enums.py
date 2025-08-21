from enum import StrEnum


class PredictionStatus(StrEnum):
    """Status of a prediction.

    >>> PredictionStatus.OK.value
    'OK'
    """

    OK = "OK"
    SKIPPED = "SKIPPED"
