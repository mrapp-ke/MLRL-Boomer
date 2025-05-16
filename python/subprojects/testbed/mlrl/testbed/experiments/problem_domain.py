"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing different kinds of problem domains.
"""
from abc import ABC, abstractmethod


class ProblemDomain(ABC):
    """
    An abstract base class for all classes that represent a specific problem domain.
    """

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


class ClassificationProblem(ProblemDomain, ABC):
    """
    Represents a classification problem.
    """

    @property
    def problem_name(self) -> str:
        """
        See :func:`mlrl.testbed.experiments.problem_domain.ProblemDomain.learner_name`
        """
        return 'classification'


class RegressionProblem(ProblemDomain, ABC):
    """
    Represents a regression problem.
    """

    @property
    def problem_name(self) -> str:
        """
        See :func:`mlrl.testbed.experiments.problem_domain.ProblemDomain.learner_name`
        """
        return 'regression'
