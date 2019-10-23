#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for inducing classification rules.
"""
import logging as log
from abc import abstractmethod
from typing import Dict

import numpy as np

from boomer.algorithm.head_refinement import HeadRefinement, SingleLabelHeadRefinement
from boomer.algorithm.losses import SquaredErrorLoss
from boomer.algorithm.model import DTYPE_SCORES, DTYPE_FEATURES, DTYPE_INDICES
from boomer.algorithm.model import Theory, Rule, EmptyBody, ConjunctiveBody, Head
from boomer.algorithm.stats import Stats, get_num_examples, get_num_features
from boomer.algorithm.sub_sampling import InstanceSubSampling, FeatureSubSampling
from boomer.learners import Module


class Refinement:

    def __init__(self, h: float, leq: bool, threshold: float, feature_index: int, threshold_index: int, head: Head,
                 covered_indices: np.ndarray):
        self.h = h
        self.leq = leq
        self.threshold = threshold
        self.feature_index = feature_index
        self.threshold_index = threshold_index
        self.head = head
        self.covered_indices = covered_indices


class RuleBuilder:
    """
    A builder that allows to configure and create a 'Rule'.

    Attributes
        leq_conditions  A 'Dict' that contains the conditions of the rule that use the "less-or-equal" operator. The
                        keys correspond to the feature indices and the values denote the thresholds to be used by the
                        conditions
        gr_conditions   A 'Dict' that contains the conditions of the rule that use the "greater" operator. The keys
                        correspond to the feature indices and the values denote the thresholds to be used by the
                        conditions
        head            The 'Head' of the rule
    """

    leq_conditions: Dict[int, float] = {}

    gr_conditions: Dict[int, float] = {}

    head: Head = None

    def apply_refinement(self, refinement: Refinement) -> 'RuleBuilder':
        """
        Applies a specific 'Refinement'.

        :param refinement:  The refinement to be applied
        :return:            The builder
        """
        self.head = refinement.head

        if refinement.leq:
            self.leq_conditions[refinement.feature_index] = refinement.threshold
        else:
            self.gr_conditions[refinement.feature_index] = refinement.threshold

        return self

    def build(self) -> Rule:
        """
        Creates and returns the rule that has been configured via the builder.

        :return: The 'Rule' that has been created
        """
        leq_features = np.fromiter(self.leq_conditions.keys(), dtype=DTYPE_INDICES)
        leq_thresholds = np.fromiter(self.leq_conditions.values(), dtype=DTYPE_FEATURES)
        gr_features = np.fromiter(self.gr_conditions.keys(), dtype=DTYPE_INDICES)
        gr_thresholds = np.fromiter(self.gr_conditions.values(), dtype=DTYPE_FEATURES)
        body = ConjunctiveBody(leq_features, leq_thresholds, gr_features, gr_thresholds)
        return Rule(body, self.head)


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

    Attributes
        presorted_indices   An array of dtype int, shape `(num_examples, num_features)`, representing the row-indices of
                            the training examples at a certain position when sorting column-wise. This matrix is cached
                            during training, if instance sub-sampling is used
    """

    presorted_indices: np.ndarray

    def __init__(self, num_rules: int = 100,
                 head_refinement: HeadRefinement = SingleLabelHeadRefinement(SquaredErrorLoss()),
                 instance_sub_sampling: InstanceSubSampling = None, feature_sub_sampling: FeatureSubSampling = None):
        """
        :param num_rules:               The number of rules to be induced (including the default rule)
        :param head_refinement:         The strategy that is used to find the heads of rules
        :param instance_sub_sampling:   The strategy that is used for sub-sampling the training examples each time a new
                                        classification rule is learned
        :param feature_sub_sampling:    The strategy that is used for sub-sampling the features each time a
                                        classification rule is refined
        """
        self.num_rules = num_rules
        self.head_refinement = head_refinement
        self.instance_sub_sampling = instance_sub_sampling
        self.feature_sub_sampling = feature_sub_sampling

    def induce_rules(self, stats: Stats, x: np.ndarray, y: np.ndarray) -> Theory:
        del self.presorted_indices
        self.__validate()

        # Convert binary ground truth labeling into expected confidence scores {-1, 1}
        expected_scores = np.ascontiguousarray(np.where(y > 0, y, -1), dtype=DTYPE_SCORES)

        # Initialize the confidence scores that are initially predicted for each example and label
        predicted_scores = np.ascontiguousarray(np.zeros(expected_scores.shape, dtype=DTYPE_SCORES))

        # Induce default rule
        log.info('Learning rule 1 / %s (default rule)...', self.num_rules)
        default_rule = self.__induce_default_rule(expected_scores, predicted_scores)

        # Apply prediction of the default rule to the matrix of predicted scores
        default_rule.predict(x, predicted_scores)

        # Create initial theory
        theory = [default_rule]

        if self.num_rules > 1:
            # Convert feature matrix into Fortran array for efficiency
            x = np.asfortranarray(x, dtype=DTYPE_FEATURES)

            while len(theory) < self.num_rules:
                log.info('Learning rule %s / %s...', len(theory) + 1, self.num_rules)
                rule = self.__induce_rule(x, expected_scores, predicted_scores)

                # Apply prediction of the new rule to the matrix of predicted scores
                rule.predict(x, predicted_scores)

                # Add new rule to theory
                theory.append(rule)
                self.random_state += 1

        del self.presorted_indices
        return theory

    def __validate(self):
        """
        Raises exceptions if the module is not configured properly.
        """

        if self.num_rules < 1:
            raise ValueError('Parameter \'num_rules\' must be at least 1, got {0}'.format(self.num_rules))
        if self.head_refinement is None:
            raise ValueError('Parameter \'head_refinement\' may not be None')

    def __induce_default_rule(self, expected_scores: np.ndarray, predicted_scores: np.ndarray) -> Rule:
        """
        Induces the default rule.

        :param expected_scores:     An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                    expected confidence scores according to the ground truth
        :param predicted_scores:    An array of dtype float, shape `(num_examples, num_labels`, representing the
                                    currently predicted confidence scores
        :return:                    The default rule
        """
        self.head_refinement.random_state = self.random_state
        head = self.head_refinement.find_default_head(expected_scores, predicted_scores)
        return Rule(EmptyBody(), head)

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
            presorted_indices = GradientBoosting.presort_features(
                grow_set) if self.presorted_indices is None else self.presorted_indices
        else:
            self.instance_sub_sampling.random_state = self.random_state
            sample_indices = self.instance_sub_sampling.sub_sample(x)
            grow_set = x[sample_indices]
            expected_scores_grow_set = expected_scores[sample_indices]
            predicted_scores_grow_set = predicted_scores[sample_indices]
            presorted_indices = None

        return self.grow_rule(grow_set, expected_scores_grow_set, predicted_scores_grow_set, presorted_indices)

    def grow_rule(self, x: np.ndarray, expected_scores: np.ndarray, predicted_scores: np.ndarray,
                  presorted_indices: np.ndarray) -> Rule:
        current_h = None
        builder = RuleBuilder()
        i = 1

        while True:
            refinement = self.__find_best_refinement(x, expected_scores, predicted_scores, presorted_indices,
                                                     iteration=i)

            if refinement is not None and (current_h is None or refinement.h < current_h):
                current_h = refinement.h
                builder.apply_refinement(refinement)
                x = x[refinement.covered_indices]
                expected_scores = expected_scores[refinement.covered_indices]
                predicted_scores = predicted_scores[refinement.covered_indices]
                i += 1
            else:
                break

        return builder.build()

    def __find_best_refinement(self, x: np.ndarray, expected_scores: np.ndarray, predicted_scores: np.ndarray,
                               presorted_indices: np.ndarray, iteration: int) -> Refinement:
        x, presorted_indices = self.__sub_sample_features(x, presorted_indices, iteration=iteration)
        x_sorted_indices = presorted_indices if presorted_indices is not None else GradientBoosting.presort_features(x)
        refinement = None

        for c in range(0, get_num_features(x)):
            indices = x_sorted_indices[:, c]
            expected_scores_sorted = expected_scores[indices]
            predicted_scores_sorted = predicted_scores[indices]

            for r in range(1, get_num_examples(x) - 1):
                current_threshold = x[indices[r - 1], c]
                next_threshold = x[indices[r], c]

                # TODO Check if the second part of the if-condition is a good idea
                if current_threshold != next_threshold and not np.array_equal(expected_scores_sorted[r - 1, :],
                                                                              expected_scores_sorted[r, :]):
                    # LEQ
                    expected_scores_subset = expected_scores_sorted[:r, :]
                    predicted_scores_subset = predicted_scores_sorted[:r, :]
                    head, h = self.head_refinement.find_head(expected_scores_subset, predicted_scores_subset)

                    if refinement is None or h <= refinement.h:
                        if refinement is None:
                            refinement = Refinement(h=h, leq=True,
                                                    threshold=GradientBoosting.__calculate_threshold(current_threshold,
                                                                                                     next_threshold),
                                                    feature_index=c, threshold_index=r, head=head,
                                                    covered_indices=indices[:r])
                        else:
                            refinement.h = h
                            refinement.leq = True
                            refinement.threshold = GradientBoosting.__calculate_threshold(current_threshold,
                                                                                          next_threshold)
                            refinement.feature_index = c
                            refinement.threshold_index = r
                            refinement.head = head
                            refinement.covered_indices = indices[:r]

                    # GR
                    expected_scores_subset = expected_scores_sorted[r:, :]
                    predicted_scores_subset = predicted_scores_sorted[r:, :]
                    head, h = self.head_refinement.find_head(expected_scores_subset, predicted_scores_subset)

                    if h < refinement.h:
                        refinement.h = h
                        refinement.leq = False
                        refinement.threshold = GradientBoosting.__calculate_threshold(current_threshold, next_threshold)
                        refinement.threshold_index = r
                        refinement.head = head
                        refinement.covered_indices = indices[r:]

        return refinement

    def __sub_sample_features(self, x: np.ndarray, presorted_indices: np.ndarray,
                              iteration: int) -> (np.ndarray, np.ndarray):
        if self.feature_sub_sampling is None:
            return x, presorted_indices
        else:
            self.feature_sub_sampling.random_state = iteration * self.random_state
            sample_indices = self.feature_sub_sampling.sub_sample(x)
            return x[sample_indices], presorted_indices[sample_indices] if presorted_indices is not None else None

    @staticmethod
    def presort_features(x: np.ndarray) -> np.ndarray:
        """
        Column-wise sorts a given feature matrix.

        :param x:   An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                    training examples
        :return:    An array of dtype int, shape `(num_examples, num_features)`, representing the row-indices of the
                    original examples at a certain position when sorting column-wise
        """
        x_sorted_indices = np.argsort(x, axis=0)
        return x_sorted_indices if x_sorted_indices.flags.fortran else np.asfortranarray(x_sorted_indices,
                                                                                         dtype=DTYPE_INDICES)

    @staticmethod
    def __calculate_threshold(first: float, second: float) -> float:
        """
        Calculates the threshold for a numerical condition as the value between two feature values.

        :param first:   The first feature value
        :param second:  The second feature value
        :return:        The threshold that has been calculated
        """
        return first + ((second - first) * 0.5)
