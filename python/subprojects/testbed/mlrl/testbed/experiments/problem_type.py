"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for dealing with different types of machine learning problems.
"""
from enum import Enum


class ProblemType(Enum):
    """
    All types of machine learning problems, an experiment may be concerned with.
    """
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
