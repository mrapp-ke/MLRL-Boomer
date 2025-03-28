"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing output data.
"""
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Dict, List, Optional, Type

from mlrl.common.config.options import Options

from mlrl.testbed.experiments.state import ExperimentState


class OutputData(ABC):
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

    @abstractmethod
    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        """
        Creates and returns a textual representation of the object.

        :param options: Options to be taken into account
        :return:        The textual representation that has been created
        """


class TabularOutputData(OutputData, ABC):
    """
    An abstract class for all classes that represent output data that can be converted into a textual, as well as a
    tabular, representation.
    """

    Table = List[Dict[str, str]]

    @abstractmethod
    def to_table(self, options: Options, **kwargs) -> Optional[Table]:
        """
        Creates and returns a tabular representation of the object.

        :param options: Options to be taken into account
        :return:        The tabular representation that has been created
        """
