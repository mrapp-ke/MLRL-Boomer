"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for differentiating different types of datasets.
"""
from enum import StrEnum


class DatasetType(StrEnum):
    """
    Characterizes a dataset as either training or test data.
    """
    TRAINING = 'training'
    TEST = 'test'
