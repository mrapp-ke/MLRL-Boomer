"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for obtaining predictions from machine learning models.
"""
import logging as log

from abc import ABC, abstractmethod
from typing import Any, Callable, Generator, Optional

from sklearn.base import ClassifierMixin, RegressorMixin

from mlrl.testbed.experiments.dataset import Dataset
from mlrl.testbed.experiments.dataset_type import DatasetType
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.state import PredictionState


class PredictionFunction:
    """
    A function that obtains and returns predictions from a learner.
    """

    def __init__(self, learner: Any, predict_function: Optional[Callable[..., Any]],
                 predict_proba_function: Optional[Callable[..., Any]]):
        """
        :param learner:                 The learner, the predictions should be obtained from
        :param predict_function:        The function to be invoked for obtaining binary predictions or scores
        :param predict_proba_function:  The function to be invoked for obtaining probability estimates
        """
        self.learner = learner
        self.predict_function = predict_function
        self.predict_proba_function = predict_proba_function

    def __predict_scores(self, dataset: Dataset, **kwargs) -> Any:
        try:
            if isinstance(self.learner, ClassifierMixin) and self.predict_function:
                return self.predict_function(dataset.x, predict_scores=True, **kwargs)
            if isinstance(self.learner, RegressorMixin) and self.predict_function:
                return self.predict_function(dataset.x, **kwargs)
            raise RuntimeError()
        except RuntimeError:
            log.error('Prediction of scores not supported')
            return None

    def __predict_probabilities(self, dataset: Dataset, **kwargs) -> Any:
        try:
            if self.predict_proba_function:
                return self.predict_proba_function(dataset.x, **kwargs)
            raise RuntimeError()
        except RuntimeError:
            log.error('Prediction of probabilities not supported')
            return None

    def __predict_binary(self, dataset: Dataset, **kwargs) -> Any:
        try:
            if self.predict_function:
                return self.predict_function(dataset.x, **kwargs)
            raise RuntimeError()
        except RuntimeError:
            log.error('Prediction of binary values not supported')
            return None

    def invoke(self, dataset: Dataset, prediction_type: PredictionType, **kwargs) -> Any:
        """
        Invokes the correct prediction function, depending on the type of the predictions that should be obtained.

        :param dataset:         The dataset that stores the query examples
        :param prediction_type: The type of the predictions that should be obtained
        :param kwargs:          Optional keyword arguments to be passed to the prediction function
        :return:                The predictions that have been obtained
        """
        if prediction_type == PredictionType.SCORES:
            return self.__predict_scores(dataset, **kwargs)
        if prediction_type == PredictionType.PROBABILITIES:
            return self.__predict_probabilities(dataset, **kwargs)
        return self.__predict_binary(dataset, **kwargs)


class Predictor(ABC):
    """
    An abstract base class for all classes that allow to obtain predictions from a previously trained model.
    """

    def __init__(self, prediction_type: PredictionType):
        """
        :param prediction_type: The type of the predictions to be obtained
        """
        self.prediction_type = prediction_type

    @abstractmethod
    def obtain_predictions(self, learner: Any, dataset: Dataset, dataset_type: DatasetType,
                           **kwargs) -> Generator[PredictionState, None, None]:
        """
        Obtains predictions from a previously trained learner once or several times.

        :param learner:         The learner
        :param dataset:         The dataset that contains the query examples
        :param dataset_type:    The type of the dataset
        :param kwargs:          Optional keyword arguments to be passed to the learner when obtaining predictions
        :return:                A generator that provides access to the results of the individual prediction processes
        """
