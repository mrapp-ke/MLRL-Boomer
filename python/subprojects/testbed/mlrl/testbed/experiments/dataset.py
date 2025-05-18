"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing datasets.
"""
from abc import ABC
from enum import Enum


class DatasetType(Enum):
    """
    Characterizes a dataset as either training or test data.
    """
    TRAINING = 'training'
    TEST = 'test'


class Dataset(ABC):
    """
    An abstract base class for all datasets.
    """
