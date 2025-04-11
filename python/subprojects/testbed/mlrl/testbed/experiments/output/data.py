"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing output data.
"""
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Iterable, List, Optional, Type

from mlrl.common.config.options import Options

from mlrl.testbed.dataset import Dataset
from mlrl.testbed.experiments.output.table import Table
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE, format_number


class OutputData(ABC):
    """
    An abstract class for all classes that represent output data that can be converted into a textual representation.
    """

    def __init__(self, name: str, file_name: str, default_context: ExperimentState.Context = ExperimentState.Context()):
        """
        :param name:            A name to be included in log messages
        :param file_name:       A file name to be used for writing into output files
        :param default_context: An `ExperimentState.Context` to be used by default for finding a suitable sink this
                                output data can be written to
        """
        self.name = name
        self.file_name = file_name
        self.default_context = default_context
        self.custom_context = {}

    def get_context(self, sink_type: Type) -> ExperimentState.Context:
        """
        Returns an `ExperimentState.Context` to can be used for finding a suitable sink of a specific type this output
        data can be written too.

        :param sink_type:   The type of the sink to be found
        :return:            An `ExperimentState.Context`
        """
        return self.custom_context.setdefault(sink_type, replace(self.default_context))

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

    @abstractmethod
    def to_table(self, options: Options, **kwargs) -> Optional[Table]:
        """
        Creates and returns a tabular representation of the object.

        :param options: Options to be taken into account
        :return:        The tabular representation that has been created
        """


class DatasetOutputData(OutputData, ABC):
    """
    An abstract base class for all classes that represent output data that can be converted into a textual
    representation, as well as a dataset.
    """

    @abstractmethod
    def to_dataset(self, options: Options, **kwargs) -> Optional[Dataset]:
        """
        Creates and returns a dataset from the object.

        :param options: Options to be taken into account
        :return:        The dataset that has been created
        """


class OutputValue:
    """
    Represents a numeric value that is part of output data.
    """

    def __init__(self, option_key: str, name: str, percentage: bool = False):
        """
        :param option_key:  The key of the option that can be used for filtering
        :param name:        The name of the value
        :param percentage:  True, if the values can be formatted as a percentage, False otherwise
        """
        self.option_key = option_key
        self.name = name
        self.percentage = percentage

    @staticmethod
    def filter_values(values: Iterable['OutputValue'], options: Options) -> List['OutputValue']:
        """
        Allows to filter given output values based on given options.

        :param values:      The output values to be filtered
        :param options:     Options that should be used for filtering
        :return:            A list that contains the filtered output values
        """
        filtered = []

        for value in values:
            option_key = value.option_key

            if options.get_bool(option_key, True):
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

        return format_number(value, decimals=kwargs.get(OPTION_DECIMALS, 0))

    def __str__(self) -> str:
        return self.name

    def __lt__(self, other: 'OutputValue') -> bool:
        return self.name < other.name

    def __hash__(self) -> int:
        return hash(self.name)
