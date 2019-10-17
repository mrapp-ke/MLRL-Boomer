#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides class for inducing classification rules.
"""
import logging as log

import numpy as np

from boomer.algorithm.boomer import RuleInduction
from boomer.algorithm.losses import Loss, SquaredErrorLoss
from boomer.algorithm.model import Theory, Rule
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
        # Convert binary ground truth labeling into expected confidence scores {-1, 1}
        ground_truth = np.where(np.nonzero(y), y, -1)
        theory = []
        t = 1

        if t <= self.num_rules:
            log.info('Learning rule %s / %s (default rule)...', t, self.num_rules)
            theory.append(self.__induce_default_rule())
            t += 1

        while t <= self.num_rules:
            log.info('Learning rule %s / %s...', t, self.num_rules)
            theory.append(self.__induce_rule())
            t += 1

        return theory

    def __induce_default_rule(self) -> Rule:
        """
        Induces the default rule.

        :return: The induced default rule
        """
        pass

    def __induce_rule(self) -> Rule:
        """
        Induces a single- or multi-label classification rule.

        :return: The induced rule
        """
        pass
