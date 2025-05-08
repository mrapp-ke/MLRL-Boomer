"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing the state of experiments.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from mlrl.testbed.experiments.dataset import Dataset, DatasetType
from mlrl.testbed.experiments.fold import Fold, FoldingStrategy
from mlrl.testbed.experiments.prediction_scope import PredictionScope
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.timer import Timer

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
        problem_type:       The type of the machine learning problem, the experiment is concerned with
        folding_strategy:   The strategy that is used for creating different folds of the dataset during the experiment
        dataset_type:       The type of the dataset used in the experiment
        dataset:            The dataset used in the experiment or None, if no dataset has been loaded yet
        fold:               The current fold of the dataset or None, if the state does not correspond to a specific fold
        parameters:         Algorithmic parameters of the learner used in the experiment
        training_result:    The result of the training process or None, if no model has been trained yet
        prediction_result:  The result of the prediction process or None, if no predictions have been obtained yet
    """
    problem_type: ProblemType
    folding_strategy: FoldingStrategy
    dataset_type: DatasetType = DatasetType.TRAINING
    dataset: Optional[Dataset] = None
    fold: Optional[Fold] = None
    parameters: ParameterDict = field(default_factory=dict)
    training_result: Optional[TrainingState] = None
    prediction_result: Optional[PredictionState] = None
