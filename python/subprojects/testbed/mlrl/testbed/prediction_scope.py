"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide information about predictions.
"""
from abc import ABC, abstractmethod
from enum import Enum

from mlrl.common.format import format_enum_values


class PredictionType(Enum):
    """
    Contains all possible types of predictions that may be obtained from a learner.
    """
    BINARY = 'binary'
    SCORES = 'scores'
    PROBABILITIES = 'probabilities'

    @staticmethod
    def parse(parameter_name: str, value: str) -> 'PredictionType':
        """
        Parses and returns a parameter value that specifies the `PredictionType` of the prediction to be obtained from a
        learner. If the given value is invalid, a `ValueError` is raised.

        :param parameter_name:  The name of the parameter
        :param value:           The value to be parsed
        :return:                A `PredictionType`
        """
        try:
            return PredictionType(value)
        except ValueError as error:
            raise ValueError('Invalid value given for parameter "' + parameter_name + '": Must be one of '
                             + format_enum_values(PredictionType) + ', but is "' + str(value) + '"') from error


class PredictionScope(ABC):
    """
    Provides information about whether predictions have been obtained from a global model or incrementally.
    """

    @abstractmethod
    def is_global(self) -> bool:
        """
        Returns whether the predictions have been obtained from a global model or not.

        :return: True, if the predictions have been obtained from a global model, False otherwise
        """

    @abstractmethod
    def get_model_size(self) -> int:
        """
        Returns the size of the model from which the prediction have been obtained.

        :return: The size of the model or 0, if the predictions have been obtained from a global model
        """

    @abstractmethod
    def get_file_name(self, name: str) -> str:
        """
        Returns a file name that corresponds to a specific prediction scope.

        :param name:    The name of the file (without suffix)
        :return:        The file name
        """


class GlobalPrediction(PredictionScope):
    """
    Provides information about predictions that have been obtained from a global model.
    """

    def is_global(self) -> bool:
        return True

    def get_model_size(self) -> int:
        return 0

    def get_file_name(self, name: str) -> str:
        return name


class IncrementalPrediction(PredictionScope):
    """
    Provides information about predictions that have been obtained incrementally.
    """

    def __init__(self, model_size: int):
        """
        :param model_size: The size of the model, the predictions have been obtained from
        """
        self.model_size = model_size

    def is_global(self) -> bool:
        return False

    def get_model_size(self) -> int:
        return self.model_size

    def get_file_name(self, name: str) -> str:
        return name + '_model-size-' + str(self.model_size)
