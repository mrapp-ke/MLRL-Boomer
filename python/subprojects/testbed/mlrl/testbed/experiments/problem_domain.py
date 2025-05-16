"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing different kinds of problem domains.
"""
from abc import ABC, abstractmethod

from mlrl.testbed.experiments.problem_type import ProblemType


class ProblemDomain(ABC):
    """
    An abstract base class for all classes that represent a specific problem domain.
    """

    def __init__(self, problem_type: ProblemType):
        """
        :param problem_type: The type of the machine learning problem
        """
        self.problem_type = problem_type

    @property
    @abstractmethod
    def problem_name(self) -> str:
        """
        The name of the problem domain.
        """

    @property
    @abstractmethod
    def learner_name(self) -> str:
        """
        The name of the machine learning algorithm used to tackle the problem domain.
        """
