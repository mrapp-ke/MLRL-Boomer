"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide information about predictions.
"""
from abc import ABC, abstractmethod
from typing import override


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


class GlobalPredictionScope(PredictionScope):
    """
    Provides information about predictions that have been obtained from a global model.
    """

    @override
    @property
    def model_size(self) -> int:
        """
        See :func:`mlrl.testbed.prediction_scope.PredictionScope.model_size`
        """
        return 0


class IncrementalPredictionScope(PredictionScope):
    """
    Provides information about predictions that have been obtained incrementally.
    """

    def __init__(self, model_size: int):
        """
        :param model_size: The size of the model, the predictions have been obtained from
        """
        self._model_size = model_size

    @override
    @property
    def model_size(self) -> int:
        """
        See :func:`mlrl.testbed.prediction_scope.PredictionScope.model_size`
        """
        return self._model_size
