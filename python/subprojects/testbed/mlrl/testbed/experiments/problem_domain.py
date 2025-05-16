"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing different kinds of problem domains.
"""
from abc import ABC, abstractmethod

from mlrl.testbed.experiments.input.dataset.splitters.splitter import DatasetSplitter
from mlrl.testbed.experiments.problem_type import ProblemType


class ProblemDomain(ABC):
    """
    An abstract base class for all classes that implement a specific problem domain.
    """

    def __init__(self, problem_type: ProblemType, dataset_splitter: DatasetSplitter):
        """
        :param problem_type:        The type of the machine learning problem
        :param dataset_splitter:    The method to be used for splitting the dataset into training and test datasets
        """
        self.problem_type = problem_type
        self.dataset_splitter = dataset_splitter

    @property
    @abstractmethod
    def learner_name(self) -> str:
        """
        The name of the machine learning algorithm used to tackle the problem domain.
        """
