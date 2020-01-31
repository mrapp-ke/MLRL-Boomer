#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement different stopping criteria that allow to decide whether additional rules should be
added to a theory or not.
"""
from abc import abstractmethod, ABC
from boomer.algorithm.model import Theory


class StoppingCriterion(ABC):
    """
    A base class for all stopping criteria that allow to decide whether additional rules should should be added to a
    theory or not.
    """

    @abstractmethod
    def should_continue(self, theory: Theory) -> bool:
        """
        Returns, whether more rules should be added to a specific theory, or not.

        :param theory:  The theory
        :return:        True, if more rules should be added to the given theory, False otherwise
        """
        pass


class SizeStoppingCriterion:
    """
    A stopping criterion that ensures that the number of rules in a theory does not exceed a certain maximum.
    """

    def __init__(self, num_rules: int):
        self.num_rules = num_rules

    def should_continue(self, theory: Theory) -> bool:
        return len(theory) < self.num_rules
