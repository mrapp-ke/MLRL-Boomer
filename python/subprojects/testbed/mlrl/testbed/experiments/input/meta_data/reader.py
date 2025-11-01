"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow reading meta-data from a source.
"""
from mlrl.testbed.experiments.input.meta_data.meta_data import InputMetaData
from mlrl.testbed.experiments.input.reader import InputReader
from mlrl.testbed.experiments.input.sources import Source


class MetaDataReader(InputReader):
    """
    Allows reading meta-data from one or several sources.
    """

    def __init__(self, *sources: Source):
        """
        :param sources: The sources, the input data should be read from
        """
        super().__init__(InputMetaData(), *sources)
