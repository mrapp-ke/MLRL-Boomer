"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for reading input data.
"""
from abc import ABC

from mlrl.testbed.experiments.connectors import Connector


class InputReader(Connector, ABC):
    """
    An abstract base class for all classes that allow to read input data.
    """
