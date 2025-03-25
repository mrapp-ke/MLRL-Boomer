"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that specify the score of output data.
"""
from dataclasses import dataclass

from mlrl.testbed.dataset import Dataset
from mlrl.testbed.fold import Fold
from mlrl.testbed.problem_type import ProblemType


@dataclass
class OutputScope:
    """
    Specifies the scope of output data.

    Attributes:
        problem_type:   The type of the machine learning problem
        dataset:        The dataset, the output data corresponds to
        fold:           The fold of the dataset, the output data corresponds to
    """
    problem_type: ProblemType
    dataset: Dataset
    fold: Fold
