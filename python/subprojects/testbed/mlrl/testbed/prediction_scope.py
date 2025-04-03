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
    def model_size(self) -> int:
        """
        The size of the model or 0, if the predictions have been obtained from a global model.
        """

    @property
    def is_global(self) -> bool:
        """
        True, if the predictions have been obtained from a global model, False otherwise.
        """
        return self.model_size == 0

    def get_file_name(self, name: str) -> str:
        """
        Returns a file name that corresponds to a specific prediction scope.

        :param name:    The name of the file (without suffix)
        :return:        The file name
        """
        return name if self.is_global else name + '_model-size-' + str(self.model_size)
