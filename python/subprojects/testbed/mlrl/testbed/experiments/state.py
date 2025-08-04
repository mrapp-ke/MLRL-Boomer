"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing the state of experiments.
"""
import logging as log

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

from mlrl.testbed.experiments.dataset import Dataset
from mlrl.testbed.experiments.dataset_type import DatasetType
from mlrl.testbed.experiments.fold import Fold, FoldingStrategy
from mlrl.testbed.experiments.meta_data import MetaData
from mlrl.testbed.experiments.prediction_scope import PredictionScope
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.problem_domain import ProblemDomain
from mlrl.testbed.experiments.timer import Timer

from mlrl.util.format import format_set

ParameterDict = Dict[str, Any]


@dataclass
class TrainingState:
    """
    Represents the result of a training process.

    Attributes:
        learner:            The learner that has been trained
        training_duration:  The time needed for training
    """
    learner: Any
    training_duration: Timer.Duration = field(default_factory=Timer.Duration)


@dataclass
class PredictionState:
    """
    Stores the result of a prediction process.

    Attributes:
        predictions:            A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray` storing predictions
        prediction_type:        The type of the predictions
        prediction_scope:       Whether the predictions have been obtained from a global model or incrementally
        prediction_duration:    The time needed for prediction
    """
    predictions: Any
    prediction_type: PredictionType
    prediction_scope: PredictionScope
    prediction_duration: Timer.Duration


@dataclass
class ExperimentState:
    """
    Represents the state of an experiment.

    Attributes:
        meta_data:          Meta-data about the command that has been used for running the experiment
        problem_domain:     The problem domain, the experiment is concerned with
        folding_strategy:   The strategy that is used for creating different folds of the dataset during the experiment
        dataset_type:       The type of the dataset used in the experiment
        dataset:            The dataset used in the experiment or None, if no dataset has been loaded yet
        fold:               The current fold of the dataset or None, if the state does not correspond to a specific fold
        parameters:         Algorithmic parameters of the learner used in the experiment
        training_result:    The result of the training process or None, if no model has been trained yet
        prediction_result:  The result of the prediction process or None, if no predictions have been obtained yet
    """
    meta_data: MetaData
    problem_domain: ProblemDomain
    folding_strategy: Optional[FoldingStrategy] = None
    dataset_type: DatasetType = DatasetType.TRAINING
    dataset: Optional[Dataset] = None
    fold: Optional[Fold] = None
    parameters: ParameterDict = field(default_factory=dict)
    training_result: Optional[TrainingState] = None
    prediction_result: Optional[PredictionState] = None

    def dataset_as(self, caller: Any, *types: Type[Dataset]) -> Optional[Dataset]:
        """
        Returns the dataset used in the experiment, if it has one of given types. Otherwise, a log message is omitted
        and `None` is returned.

        :param caller:  The caller of this function to be included in the log message
        :param types:   The accepted types
        :return:        The dataset or None, if it does not have the correct type
        """
        dataset = self.dataset

        if any(isinstance(dataset, dataset_type) for dataset_type in types):
            return dataset

        log.error('%s expected type of dataset to be one of %s, but dataset has type %s',
                  type(caller).__qualname__, format_set(map(lambda dataset_type: dataset_type.__name__, types)),
                  type(dataset).__name__)
        return None

    def learner_as(self, caller: Any, *types: Type[Any]) -> Optional[Any]:
        """
        Returns the learner that has been trained in the experiment, if it has one of given types. Otherwise, a log
        message is omitted and `None` is returned.

        :param caller:  The caller of this function to be included in the log message
        :param types:   The accepted types
        :return:        The learner or None, if it does not have the correct type
        """
        learner = None
        training_result = self.training_result

        if training_result:
            learner = training_result.learner

            if any(isinstance(learner, learner_type) for learner_type in types):
                return learner

        log.error('%s expected type of learner to be one of %s, but learner has type %s',
                  type(caller).__qualname__, format_set(map(lambda learner_type: learner_type.__name__, types)),
                  type(learner).__name__)
        return None
