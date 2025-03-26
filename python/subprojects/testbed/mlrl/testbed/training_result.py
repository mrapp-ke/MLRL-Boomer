"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that store the result of a training process.
"""
from dataclasses import dataclass

from sklearn.base import BaseEstimator


@dataclass
class TrainingResult:
    """
    Stores the result of a training process.

    Attributes:
        learner:    The learner that has been trained
        train_time: The time needed for training
    """
    learner: BaseEstimator
    train_time: float
