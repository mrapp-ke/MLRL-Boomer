"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow reading models from one or several sources.
"""
from mlrl.testbed.experiments.input.model.model import InputModel
from mlrl.testbed.experiments.input.reader import InputReader
from mlrl.testbed.experiments.input.sources import Source


class ModelReader(InputReader):
    """
    Allows reading a model from one or several sources.
    """

    def __init__(self, *sources: Source):
        """
        :param source: The sources, the input data should be read from
        """
        super().__init__(InputModel(), *sources)
