"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing the state of experiments.
"""
from dataclasses import dataclass
from typing import Any, Dict

from mlrl.testbed.dataset import Dataset
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.fold import Fold


@dataclass
class ExperimentState:
    """
    Represents the state of an experiment.

    Attributes:
        problem_type:   The type of the machine learning problem, the experiment is concerned with
        dataset:        The dataset used in the experiment
        fold:           The current fold of the dataset used in the experiment
        parameters:     Algorithmic parameters of the learner used in the experiment
    """
    problem_type: ProblemType
    dataset: Dataset
    fold: Fold
    parameters: Dict[str, Any]
