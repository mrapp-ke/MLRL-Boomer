"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for reading input data.
"""
from abc import ABC

from mlrl.testbed.experiments.data import DataExchange


class InputReader(DataExchange, ABC):
    """
    An abstract base class for all classes that allow to read input data.
    """
