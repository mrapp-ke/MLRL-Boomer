#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides class for inducing classification rules.
"""
import logging as log
from abc import abstractmethod

import numpy as np

from boomer.algorithm.losses import Loss, DecomposableLoss, SquaredErrorLoss
from boomer.algorithm.model import Theory, Rule, EmptyBody, Head, DTYPE_SCORES, DTYPE_FEATURES, DTYPE_INDICES
from boomer.algorithm.rule_refinement import refine_rule, AllFeaturesIterator
from boomer.algorithm.stats import Stats
from boomer.algorithm.sub_sampling import InstanceSubSampling, Bagging
from boomer.learners import Module


class RuleInduction(Module):
    """
    A module that allows to induce a `Theory`, consisting of several classification rules.
    """

    @abstractmethod
    def induce_rules(self, stats: Stats, x: np.ndarray, y: np.ndarray) -> Theory:
        """
        Creates and returns a 'Theory' that contains several candidate rules.

        :param stats:   Statistics about the training data set
        :param x:       An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                        training examples
        :param y:       An array of dtype float, shape `(num_examples, num_labels)`, representing the labels of the
                        training examples
        :return:        A 'Theory' that contains the generated candidate rules
        """
        pass


class GradientBoosting(RuleInduction):
    """
    Implements the induction of (multi-label) classification rules using gradient boosting.
    """

    def __init__(self, num_rules: int = 100, loss: Loss = SquaredErrorLoss(),
                 instance_sub_sampling: InstanceSubSampling = Bagging()):
        """
        :param num_rules:               The number of rules to be induced (including the default rule)
        :param loss:                    The (surrogate) loss to be minimized
        :param instance_sub_sampling:   The strategy that is used for sub-sampling the training examples each time a new
                                        classification rule is learned
        """
        self.num_rules = num_rules
        self.loss = loss
        self.instance_sub_sampling = instance_sub_sampling

    def induce_rules(self, stats: Stats, x: np.ndarray, y: np.ndarray) -> Theory:
        self.__validate()

        # Convert binary ground truth labeling into expected confidence scores {-1, 1}
        expected_scores = np.ascontiguousarray(np.where(y > 0, y, -1), dtype=DTYPE_SCORES)

        # Initialize the confidence scores that are initially predicted for each example and label
        predicted_scores = np.ascontiguousarray(np.zeros(expected_scores.shape, dtype=DTYPE_SCORES))

        # Induce default rule
        log.info('Learning rule 1 / %s (default rule)...', self.num_rules)
        default_rule, predicted_scores = self.__induce_default_rule(expected_scores, predicted_scores)
        theory = [default_rule]

        if self.num_rules > 1:
            # Convert feature matrix into Fortran array for efficiency
            x = np.asfortranarray(x, dtype=DTYPE_FEATURES)

            while len(theory) < self.num_rules:
                log.info('Learning rule %s / %s...', len(theory) + 1, self.num_rules)
                rule = self.__induce_rule(x, expected_scores, predicted_scores)

                # TODO: Can we make this more efficient?
                mask = rule.body.match(x)
                predicted_scores[mask] += rule.head

                theory.append(rule)
                self.random_state += 1

        return theory

    def __validate(self):
        """
        Raises exceptions if the module is not configured properly.
        """

        if self.num_rules < 1:
            raise ValueError('Parameter \'num_rules\' must be at least 1, got {0}'.format(self.num_rules))
        if self.loss is None:
            raise ValueError('Parameter \'loss\' may not be None')

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

        if isinstance(self.loss, DecomposableLoss):
            head, predicted_scores = self.__derive_full_head_using_decomposable_loss(expected_scores, predicted_scores)
        else:
            # TODO: Implement
            raise NotImplementedError('Non-decomposable loss functions not supported yet...')

        return Rule(EmptyBody(), head), predicted_scores

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

        head, _ = self.loss.derive_scores(expected_scores, predicted_scores)
        predicted_scores += head
        return head, predicted_scores

    def __induce_rule(self, x: np.ndarray, expected_scores: np.ndarray, predicted_scores: np.ndarray) -> Rule:
        """
        Induces a single- or multi-label classification rule.

        :param x:   An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                    training examples
        :return:    The induced rule
        """

        if self.instance_sub_sampling is None:
            grow_set = x
            expected_scores_grow_set = expected_scores
            predicted_scores_grow_set = predicted_scores
        else:
            self.instance_sub_sampling.random_state = self.random_state
            sample_indices = self.instance_sub_sampling.sub_sample(x)
            grow_set = x[sample_indices]
            expected_scores_grow_set = expected_scores[sample_indices]
            predicted_scores_grow_set = predicted_scores[sample_indices]

        # Presort features for efficient evaluation of numerical conditions
        # TODO Keep if no sub-sampling is used
        x_sorted_indices = np.asfortranarray(np.argsort(grow_set, axis=0), dtype=DTYPE_INDICES)

        return refine_rule(grow_set, x_sorted_indices, expected_scores_grow_set, predicted_scores_grow_set, self.loss,
                           AllFeaturesIterator(grow_set))
