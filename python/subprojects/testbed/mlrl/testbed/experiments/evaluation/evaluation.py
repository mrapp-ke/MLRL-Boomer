"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for evaluating predictions provided by a machine learning model.
"""
import logging as log

from abc import ABC, abstractmethod
from typing import List

from sklearn.base import RegressorMixin

from mlrl.common.mixins import ClassifierMixin

from mlrl.testbed.dataset import Dataset
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.prediction_scope import PredictionType


class Evaluation(ABC):
    """
    An abstract base class for all classes that allow to evaluate predictions provided by a previously trained model.
    """

    def __init__(self, prediction_type: PredictionType, output_writers: List[OutputWriter]):
        """
        :param prediction_type: The type of the predictions to be obtained
        :param output_writers:  A list that contains all output writers to be invoked after predictions have been
                                obtained
        """
        self.prediction_type = prediction_type
        self.output_writers = output_writers

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

    def _evaluate_predictions(self, state: ExperimentState):
        """
        May be used by subclasses in order to evaluate predictions that have been obtained from a previously trained
        model.

        :param state: The state that stores the predictions and the model
        """
        for output_writer in self.output_writers:
            output_writer.write_output(state)

    @abstractmethod
    def predict_and_evaluate(self, state: ExperimentState, **kwargs):
        """
        Must be implemented by subclasses in order to obtain and evaluate predictions from a previously trained model.

        :param state:   The state that stores the model
        :param kwargs:  Optional keyword arguments to be passed to the model when obtaining predictions
        """
