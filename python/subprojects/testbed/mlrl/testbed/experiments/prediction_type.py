"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for dealing with different types of predictions.
"""
from enum import StrEnum


class PredictionType(StrEnum):
    """
    Specifies all possible types of predictions that may be obtained from a machine learning model.
    """
    BINARY = 'binary'
    SCORES = 'scores'
    PROBABILITIES = 'probabilities'
