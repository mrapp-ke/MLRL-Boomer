#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides class for inducing classification rules.
"""
import logging as log

import numpy as np

from boomer.algorithm.boomer import RuleInduction
from boomer.algorithm.losses import Loss, DecomposableLoss, SquaredErrorLoss
from boomer.algorithm.model import Theory, Rule, EmptyBody, Head
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
        expected_scores = np.where(np.nonzero(y), y, -1)
        # Initialize the confidence scores that are initially predicted for each example and label
        predicted_scores = np.zeros(np.shape(expected_scores), dtype=float)
        theory = []
        t = 1

        if t <= self.num_rules:
            log.info('Learning rule %s / %s (default rule)...', t, self.num_rules)
            default_rule, predicted_scores = self.__induce_default_rule(expected_scores, predicted_scores)
            theory.append(default_rule)
            t += 1

        while t <= self.num_rules:
            log.info('Learning rule %s / %s...', t, self.num_rules)
            # TODO theory.append(self.__induce_rule())
            t += 1

        return theory

    def __induce_default_rule(self, expected_scores: np.ndarray, predicted_scores: np.ndarray) -> (Rule, np.ndarray):
        """
        Induces the default rule.

        :param expected_scores:     An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                    expected confidence scores according to the ground truth
        :param predicted_scores:    An array of dtype float, shape `(num_examples, num_labels`, representing the
                                    currently predicted confidence scores
        :return:                    The induced default rule, as well as the confidence scores that are predicted after
                                    including the rule in the theory
        """

        if self.loss is DecomposableLoss:
            head, predicted_scores = self.__derive_full_head_using_decomposable_loss(expected_scores, predicted_scores)
        else:
            # TODO: Implement
            pass

        rule = Rule(EmptyBody(), head)
        return rule, predicted_scores

    def __derive_full_head_using_decomposable_loss(self, expected_scores: np.ndarray,
                                                   predicted_scores: np.ndarray) -> (Head, np.ndarray):
        """
        Derives a full head, if the used loss function is decomposable.

        :param expected_scores:     An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                    expected confidence scores according to the ground truth
        :param predicted_scores:    An array of dtype float, shape `(num_examples, num_labels`, representing the
                                    currently predicted confidence scores
        :return:                    The derived head, as well as the confidence scores that predicted after applying the
                                    head to the given predicted scores
        """

        head = self.loss.derive_scores(expected_scores, predicted_scores)
        predicted_scores += head
        return head, predicted_scores

    def __induce_rule(self) -> Rule:
        """
        Induces a single- or multi-label classification rule.

        :return: The induced rule
        """
        pass
