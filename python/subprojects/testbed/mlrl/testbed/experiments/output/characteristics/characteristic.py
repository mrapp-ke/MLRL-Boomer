"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing individual characteristics that are part of output data.
"""
from numbers import Number
from typing import Any, Callable

from mlrl.testbed.experiments.output.data import OutputValue


class Characteristic(OutputValue):
    """
    An individual characteristic that is part of output data.
    """

    Function = Callable[[Any], Number]

    def __init__(self, option_key: str, name: str, function: Function, percentage: bool = False):
        """
        :param function: The function that should be used to retrieve the value of the characteristic
        """
        super().__init__(option_key=option_key, name=name, percentage=percentage)
        self.function = function

    def format(self, value, **kwargs) -> str:
        """
        See :func:`mlrl.testbed.experiments.data.OutputValue.format`
        """
        return super().format(self.function(value), **kwargs)
