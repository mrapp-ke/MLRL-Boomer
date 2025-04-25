"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing input or output data.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from os import path
from typing import Optional, Type

from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.io import get_file_name_per_fold


class Data(ABC):
    """
    An abstract class for all classes that represent data that can be exchanged with the environment, this software runs
    in.
    """

    @dataclass
    class Context:
        """
        Specifies the aspects of an `ExperimentState` that should be taken into account for finding a suitable data
        exchange for handling data.

        Attributes:
            include_dataset_type:       True, if the type of the dataset should be taken into account, False otherwise
            include_prediction_scope:   True, if the scope of predictions should be taken into account, False otherwise
            include_fold:               True, if the cross validation fold should be taken into account, False otherwise
        """
        include_dataset_type: bool = True
        include_prediction_scope: bool = True
        include_fold: bool = True

    def __init__(self, default_context: Context = Context()):
        """
        :param default_context: A `Data.Context` to be used by default for finding a suitable data exchange this data
                                can be handled by
        """
        self.default_context = default_context
        self.custom_context = {}

    def get_context(self, exchange_type: Type) -> Context:
        """
        Returns a `Data.Context` to can be used for finding a suitable data exchange for handling this data.

        :param exchange_type:   The type of the data exchange to search for
        :return:                A `Data.Context`
        """
        return self.custom_context.setdefault(exchange_type, replace(self.default_context))


class DataExchange(ABC):
    """
    An abstract base class for all classes that allow to exchange data with the environment this software runs in.
    """

    class Session(ABC):
        """
        An abstract base class for all sessions of a `DataExchange` that can be used for exchanging data with the
        environment this software runs in.
        """

        @abstractmethod
        def exchange(self) -> Optional[Data]:
            """
            Exchanges data with the environment.

            :return: Data that has been received from the environment or None, if no data has been received
            """

    @abstractmethod
    def open_session(self, state: ExperimentState) -> Session:
        """
        Opens a session for exchanging data with the environment.

        :param state:   The current state of the experiment
        :return:        The session that has been opened
        """


class FilePath:
    """
    The path to a file to exchange data with.
    """

    def __init__(self, directory: str, file_name: str, suffix: str, context: Data.Context):
        """
        :param directory:   The path to the directory, where the file is located
        :param file_name:   The name of the file
        :param suffix:      The suffix of the file (with leading dot)
        :param context:     A `Data.Context` to be used to determine the path
        """
        self.directory = directory
        self.file_name = file_name
        self.suffix = suffix
        self.context = context

    def resolve(self, state: ExperimentState) -> str:
        """
        Determines and returns the path to the file to which output data should be written.

        :param state: The state from which the output data has been generated
        """
        file_name = self.file_name

        if self.context.include_dataset_type:
            file_name = state.dataset.type.get_file_name(file_name)

        if self.context.include_prediction_scope:
            prediction_result = state.prediction_result

            if prediction_result:
                file_name = prediction_result.prediction_scope.get_file_name(file_name)

        if self.context.include_fold:
            fold = state.fold if state.folding_strategy.is_cross_validation_used else None
            file_name = get_file_name_per_fold(file_name, self.suffix, fold)

        return path.join(self.directory, file_name)
