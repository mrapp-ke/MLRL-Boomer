"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing output data.
"""
from abc import ABC

from mlrl.common.config.options import Options

from mlrl.testbed.experiments.output.converters import TableConverter, TextConverter
from mlrl.testbed.experiments.state import ExperimentState


class OutputData(TextConverter, TableConverter, ABC):
    """
    An abstract base class for all classes that represent output data.
    """

    def __init__(self,
                 name: str,
                 file_name: str,
                 formatter_options: ExperimentState.FormatterOptions = ExperimentState.FormatterOptions()):
        """
        :param name:                A name to be included in log messages
        :param file_name:           A file name to be used for writing into output files
        :param formatter_options:   The options to be used for creating textual representations of the
                                    `ExperimentState`, the output data has been generated from
        """
        self.name = name
        self.file_name = file_name
        self.formatter_options = formatter_options
