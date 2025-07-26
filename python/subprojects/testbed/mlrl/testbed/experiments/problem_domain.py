"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing different kinds of problem domains.
"""
from abc import ABC, abstractmethod
from typing import override


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

    NAME = 'classification'

    @override
    @property
    def problem_name(self) -> str:
        """
        See :func:`mlrl.testbed.experiments.problem_domain.ProblemDomain.learner_name`
        """
        return self.NAME


class RegressionProblem(ProblemDomain, ABC):
    """
    Represents a regression problem.
    """

    NAME = 'regression'

    @override
    @property
    def problem_name(self) -> str:
        """
        See :func:`mlrl.testbed.experiments.problem_domain.ProblemDomain.learner_name`
        """
        return self.NAME
