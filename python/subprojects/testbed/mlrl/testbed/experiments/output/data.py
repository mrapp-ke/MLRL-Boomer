"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing output data.
"""
import json

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Type, override

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.data import Properties, TabularProperties
from mlrl.testbed.experiments.dataset import Dataset
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.experiments.table import Table
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE, format_number

from mlrl.util.options import Options


class OutputData(ABC):
    """
    An abstract class for all classes that represent output data.
    """

    def __init__(self, properties: Properties, context: Context = Context()):
        """
        :param properties:  The properties of the output data
        :param context:     A `Context` to be used by default for finding a suitable sink this output data can be
                            written to
        """
        self.properties = properties
        self.context = context
        self.custom_context: Dict[Type[Any], Context] = {}

    def get_context(self, lookup_type: Type[Any]) -> Context:
        """
        Returns a `Context` that can be used for finding a suitable sink for handling this data.

        :param lookup_type: The type of the sink to search for
        :return:            A `Context`
        """
        return self.custom_context.setdefault(lookup_type, replace(self.context))


class TextualOutputData(OutputData, ABC):
    """
    An abstract class for all classes that represent output data that can be converted into a textual representation.
    """

    class Title:
        """
        A title that is printed before textual output data.
        """

        def __init__(self, title: str, context: Context):
            """
            :param title:   A title
            :param context: A `Context` to be used for formatting the title
            """
            self.title = title
            self.context = context

        def __format_dataset_type(self, state: ExperimentState) -> str:
            if self.context.include_dataset_type:
                dataset_type = state.dataset_type

                if dataset_type:
                    return ' for ' + dataset_type + ' data'

            return ''

        def __format_fold(self, state: ExperimentState) -> str:
            if self.context.include_fold:
                folding_strategy = state.folding_strategy

                if folding_strategy and folding_strategy.is_cross_validation_used:
                    fold = state.fold

                    if fold:
                        formatted_fold = 'Fold ' + str(fold.index + 1)
                    else:
                        formatted_fold = 'Average across ' + str(folding_strategy.num_folds) + ' folds'

                    return ' (' + formatted_fold + ')'

            return ''

        def __format_prediction_scope(self, state: ExperimentState) -> str:
            if self.context.include_prediction_scope:
                prediction_result = state.prediction_result

                if prediction_result:
                    prediction_scope = prediction_result.prediction_scope

                    if not prediction_scope.is_global:
                        return ' using a model of size ' + str(prediction_scope.model_size)

            return ''

        def format(self, state: ExperimentState) -> str:
            """
            Formats and returns the title that is printed before the output data.

            :param state: The state from which the output data has been generated
            """
            return self.title + self.__format_dataset_type(state) + self.__format_prediction_scope(
                state) + self.__format_fold(state)

    @staticmethod
    def from_text(properties: Properties, context: Context, text: str) -> 'TextualOutputData':
        """
        Creates and returns `TextualOutputData` from a given text.

        :param properties:  The properties to be used
        :param context:     The context to be used
        :param text:        The text to be used
        :return:            The `TextualOutputData` that has been created
        """

        class TextOutputData(TextualOutputData):
            """
            Output data that consists of a text.
            """

            @override
            def to_text(self, options: Options, **kwargs) -> Optional[str]:
                return text

        return TextOutputData(properties=properties, context=context)

    @abstractmethod
    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        """
        Creates and returns a textual representation of the object.

        :param options: Options to be taken into account
        :return:        The textual representation that has been created
        """


class TabularOutputData(TextualOutputData, ABC):
    """
    An abstract class for all classes that represent output data that can be converted into a textual, as well as a
    tabular, representation.
    """

    def __init__(self, properties: TabularProperties, context: Context = Context()):
        """
        :param properties:  The properties of the output data
        :param context:     A `Context` to be used by default for finding a suitable sink this output data can be
                            written to
        """
        super().__init__(properties=properties, context=context)

    @staticmethod
    def from_table(properties: TabularProperties, context: Context, table: Table) -> 'TabularOutputData':
        """
        Creates and returns `TabularOutputData` from a given table.

        :param properties:  The properties to be used
        :param context:     The context to be used
        :param table:       The table to be used
        :return:            The `TabularOutputData` that has been created
        """

        class TableOutputData(TabularOutputData):
            """
            Output data that consists of a table.
            """

            @override
            def to_text(self, options: Options, **kwargs) -> Optional[str]:
                return table.format()

            @override
            def to_table(self, options: Options, **kwargs) -> Optional[Table]:
                return table

        return TableOutputData(properties=properties, context=context)

    @abstractmethod
    def to_table(self, options: Options, **kwargs) -> Optional[Table]:
        """
        Creates and returns a tabular representation of the object.

        :param options: Options to be taken into account
        :return:        The tabular representation that has been created
        """


class StructuralOutputData(TextualOutputData, ABC):
    """
    An abstract base class for all classes that represent output data that can be converted into a structural
    representation, e.g., YAML or JSON.
    """

    @abstractmethod
    def to_dict(self, options: Options, **kwargs) -> Optional[Dict[Any, Any]]:
        """
        Creates and returns a dictionary from the object.

        :param options: Options to be taken into account
        :return:        The dictionary that has been created
        """

    @override
    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        dictionary = self.to_dict(options, **kwargs)
        return None if dictionary is None else json.dumps(dictionary, indent=4)


class DatasetOutputData(TextualOutputData, ABC):
    """
    An abstract base class for all classes that represent output data that can be converted into a textual
    representation, as well as a dataset.
    """

    @abstractmethod
    def to_dataset(self, options: Options, **kwargs) -> Optional[Dataset]:
        """
        Creates and returns a dataset from the object.

        :param options: Options to be taken into account
        :return:        The dataset
        """


class ObjectOutputData(OutputData, ABC):
    """
    An abstract base class for all classes that represent output data that can be converted into a Python object.
    """

    @abstractmethod
    def to_object(self, options: Options, **kwargs) -> Optional[Any]:
        """
        Returns an object.

        :param options: Options to be taken into account
        :return:        The object
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

    def std_dev(self) -> 'OutputValue':
        """
        Creates and returns an `OutputValue` that corresponds to the standard deviation of this value.

        :return: An `OutputValue` that corresponds to the standard deviation of this value
        """
        return OutputValue(option_key=self.option_key, name='Std.-dev. ' + self.name, percentage=self.percentage)

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

    @override
    def __str__(self) -> str:
        return self.name

    def __lt__(self, other: 'OutputValue') -> bool:
        return self.name < other.name

    @override
    def __hash__(self) -> int:
        return hash(self.name)
