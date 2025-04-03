"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide information about predictions.
"""
from abc import ABC, abstractmethod


class PredictionScope(ABC):
    """
    Provides information about whether predictions have been obtained from a global model or incrementally.
    """

    @property
    @abstractmethod
    def is_global(self) -> bool:
        """
        True, if the predictions have been obtained from a global model, False otherwise.
        """

    @property
    @abstractmethod
    def model_size(self) -> int:
        """
        The size of the model or 0, if the predictions have been obtained from a global model.
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

    @property
    def is_global(self) -> bool:
        return True

    @property
    def model_size(self) -> int:
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
        self._model_size = model_size

    @property
    def is_global(self) -> bool:
        return False

    @property
    def model_size(self) -> int:
        return self._model_size

    def get_file_name(self, name: str) -> str:
        return name + '_model-size-' + str(self.model_size)
