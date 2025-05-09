"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow reading models from a source.
"""
from mlrl.testbed.experiments.input.model.model import InputModel
from mlrl.testbed.experiments.input.reader import InputReader
from mlrl.testbed.experiments.input.sources import Source


class ModelReader(InputReader):
    """
    Allows reading a model from a source.
    """

    def __init__(self, source: Source):
        """
        :param source: The source, the input data should be read from
        """
        super().__init__(source, InputModel())
