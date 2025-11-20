"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for obtaining predictions from global machine learning models.
"""
import logging as log

from typing import Any, Generator, override

from mlrl.testbed_sklearn.experiments.prediction.predictor import PredictionFunction, Predictor

from mlrl.testbed.experiments.dataset import Dataset
from mlrl.testbed.experiments.dataset_type import DatasetType
from mlrl.testbed.experiments.prediction_scope import GlobalPredictionScope
from mlrl.testbed.experiments.state import PredictionResult, PredictionState
from mlrl.testbed.experiments.timer import Timer


class GlobalPredictionFunction(PredictionFunction):
    """
    A function that obtains and returns global predictions from a learner.
    """

    def __init__(self, learner: Any):
        """
        :param learner: The learner, the predictions should be obtained from
        """
        super().__init__(
            learner=learner,
            predict_function=learner.predict,
            decision_function=learner.decision_function
            if callable(getattr(learner, 'decision_function', None)) else None,
            predict_proba_function=learner.predict_proba if callable(getattr(learner, 'predict_proba', None)) else None)


class GlobalPredictor(Predictor):
    """
    Obtains predictions from a previously trained global model.
    """

    @override
    def obtain_predictions(self, learner: Any, dataset: Dataset, dataset_type: DatasetType,
                           **kwargs) -> Generator[PredictionState, None, None]:
        """
        See :func:`mlrl.testbed_sklearn.experiments.prediction.predictor.Predictor.obtain_predictions`
        """
        log.info('Predicting for %s %s examples...', dataset.num_examples, dataset_type)
        start_time = Timer.start()
        prediction_function = GlobalPredictionFunction(learner)
        predictions = prediction_function.invoke(dataset, self.prediction_type, **kwargs)
        prediction_duration = Timer.stop(start_time)

        if predictions is not None:
            log.info('Successfully predicted in %s', prediction_duration)
            yield PredictionState(prediction_scope=GlobalPredictionScope(),
                                  prediction_result=PredictionResult(predictions=predictions,
                                                                     prediction_type=self.prediction_type,
                                                                     prediction_duration=prediction_duration))
