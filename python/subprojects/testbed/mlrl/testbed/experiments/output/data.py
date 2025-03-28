"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing output data.
"""
from abc import ABC
from dataclasses import replace
from typing import Type

from mlrl.testbed.experiments.output.converters import TableConverter, TextConverter
from mlrl.testbed.experiments.state import ExperimentState


class OutputData(TextConverter, ABC):
    """
    An abstract class for all classes that represent output data that can be converted into a textual representation.
    """

    def __init__(self,
                 name: str,
                 file_name: str,
                 default_formatter_options: ExperimentState.FormatterOptions = ExperimentState.FormatterOptions()):
        """
        :param name:                        A name to be included in log messages
        :param file_name:                   A file name to be used for writing into output files
        :param default_formatter_options:   The options to be used for creating textual representations of the
                                            `ExperimentState`, the output data has been generated from
        """
        self.name = name
        self.file_name = file_name
        self.default_formatter_options = default_formatter_options
        self.custom_formatter_options = {}

    def get_formatter_options(self, sink_type: Type) -> ExperimentState.FormatterOptions:
        """
        Returns the options to be used by a specific type of sink for creating textual representation of an
        `ExperimentState`.

        :param sink_type:   The type of the sink
        :return:            The options to be used by the given type of sink
        """
        return self.custom_formatter_options.setdefault(sink_type, replace(self.default_formatter_options))


class TabularOutputData(OutputData, TableConverter, ABC):
    """
    An abstract class for all classes that represent output data that can be converted into a textual, as well as a
    tabular, representation.
    """
