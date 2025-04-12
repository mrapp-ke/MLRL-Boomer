"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing the state of experiments.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

from sklearn.base import BaseEstimator

from mlrl.testbed.dataset import Dataset
from mlrl.testbed.experiments.prediction_scope import PredictionScope
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.timer import Timer
from mlrl.testbed.fold import Fold

ParameterDict = Dict[str, Any]


@dataclass
class TrainingState:
    """
    Represents the result of a training process.

    Attributes:
        learner:            The learner that has been trained
        training_duration:  The time needed for training
    """
    learner: BaseEstimator
    training_duration: Timer.Duration


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
        problem_type:       The type of the machine learning problem, the experiment is concerned with
        dataset:            The dataset used in the experiment
        fold:               The current fold of the dataset used in the experiment
        parameters:         Algorithmic parameters of the learner used in the experiment
        training_result:    The result of the training process or None, if no model has been trained yet
        prediction_result:  The result of the prediction process or None, if no predictions have been obtained yet
    """
    problem_type: ProblemType
    dataset: Dataset
    fold: Fold
    parameters: ParameterDict
    training_result: Optional[TrainingState] = None
    prediction_result: Optional[PredictionState] = None
