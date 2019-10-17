#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides class for inducing classification rules.
"""
import numpy as np

from boomer.algorithm.boomer import RuleInduction
from boomer.algorithm.losses import Loss, SquaredErrorLoss
from boomer.algorithm.model import Theory
from boomer.algorithm.stats import Stats


class GradientBoosting(RuleInduction):
    """
    Implements the induction of (multi-label) classification rules using gradient boosting.
    """

    def __init__(self, num_rules: int = 100, loss: Loss = SquaredErrorLoss()):
        """
        :param num_rules:   The number of rules to be induced (including the default rule)
        :param loss:        The (surrogate) loss to be minimized
        """
        self.num_rules = num_rules
        self.loss = loss

    def induce_rules(self, stats: Stats, x: np.ndarray, y: np.ndarray) -> Theory:
        pass
