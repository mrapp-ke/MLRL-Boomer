"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing output data.
"""
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Type

from mlrl.common.config.options import Options

from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE, format_float


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


class OutputValue:
    """
    Represents a numeric value that is part of output data.
    """

    def __init__(self, option_key: str, name: str, percentage: bool = False):
        """
        :param option_key:  The key of the option that can be used for filtering
        :param name:        A name that describes the type of values
        :param percentage:  True, if the values can be formatted as a percentage, False otherwise
        """
        self.option_key = option_key
        self.name = name
        self.percentage = percentage

    @staticmethod
    def filter_values(values: Iterable['OutputValue'], *options: Options) -> List['OutputValue']:
        """
        Allows to filter given output values based on given options.

        :param values:      The output values to be filtered
        :param options:     The options that should be used for filtering
        :return:            A list that contains the filtered output values
        """
        filtered = []

        for value in values:
            option_key = value.option_key

            if any(current_options.get_bool(option_key, True) for current_options in options):
                filtered.append(value)

        return filtered

    def format(self, value, **kwargs) -> str:
        """
        Creates and returns a textual representation of a given value.

        :param value:   The value
        :return:        The textual representation that has been created
        """
        if self.percentage and kwargs.get(OPTION_PERCENTAGE, False):
            value = value * 100

        return format_float(value, decimals=kwargs.get(OPTION_DECIMALS, 0))

    def __str__(self) -> str:
        return self.name

    def __lt__(self, other: 'OutputValue') -> bool:
        return self.name < other.name

    def __hash__(self) -> int:
        return hash(self.name)
