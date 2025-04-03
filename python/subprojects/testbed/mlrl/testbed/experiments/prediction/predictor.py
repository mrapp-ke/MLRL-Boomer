"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for obtaining predictions from machine learning models.
"""
import logging as log

from abc import ABC, abstractmethod
from typing import Generator

from sklearn.base import RegressorMixin

from mlrl.common.mixins import ClassifierMixin

from mlrl.testbed.dataset import Dataset
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.state import ExperimentState, PredictionState


class Predictor(ABC):
    """
    An abstract base class for all classes that allow to obtain predictions from a previously trained model.
    """

    def __init__(self, prediction_type: PredictionType):
        """
        :param prediction_type: The type of the predictions to be obtained
        """
        self.prediction_type = prediction_type

    def _invoke_prediction_function(self, learner, predict_function, predict_proba_function, dataset: Dataset,
                                    **kwargs):
        """
        May be used by subclasses in order to invoke the correct prediction function, depending on the type of
        result that should be obtained.

        :param learner:                 The learner, the result should be obtained from
        :param predict_function:        The function to be invoked if binary results or scores should be obtained
        :param predict_proba_function:  The function to be invoked if probability estimates should be obtained
        :param dataset:                 The dataset that stores the query examples
        :param kwargs:                  Optional keyword arguments to be passed to the `predict_function`
        :return:                        The return value of the invoked function
        """
        prediction_type = self.prediction_type
        x = dataset.x

        if prediction_type == PredictionType.SCORES:
            try:
                if isinstance(learner, ClassifierMixin):
                    result = predict_function(x, predict_scores=True, **kwargs)
                elif isinstance(learner, RegressorMixin):
                    result = predict_function(x, **kwargs)
                else:
                    raise RuntimeError()
            except RuntimeError:
                log.error('Prediction of scores not supported')
                result = None
        elif prediction_type == PredictionType.PROBABILITIES:
            try:
                result = predict_proba_function(x)
            except RuntimeError:
                log.error('Prediction of probabilities not supported')
                result = None
        else:
            result = predict_function(x, **kwargs)

        return result

    @abstractmethod
    def obtain_predictions(self, state: ExperimentState, **kwargs) -> Generator[PredictionState]:
        """
        Obtains predictions from a previously trained model once or several times.

        :param state:   The state that stores the model
        :param kwargs:  Optional keyword arguments to be passed to the model when obtaining predictions
        :return:        A generator that provides access to the results of the individual prediction processes
        """
