"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow reading algorithmic parameters from one or several sources.
"""

from mlrl.testbed.experiments.input.parameters.parameters import InputParameters
from mlrl.testbed.experiments.input.reader import InputReader
from mlrl.testbed.experiments.input.sources import Source


class ParameterReader(InputReader):
    """
    Allows reading algorithmic parameters from one or several sources.
    """

    def __init__(self, *sources: Source):
        """
        :param sources: The sources, the input data should be read from
        """
        super().__init__(InputParameters(), *sources)
