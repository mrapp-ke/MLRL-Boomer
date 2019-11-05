#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for inducing classification rules.
"""
import logging as log
from abc import abstractmethod
from typing import Dict

import numpy as np
from boomer.algorithm._losses import SquaredErrorLoss
from boomer.algorithm._model import Rule, EmptyBody, ConjunctiveBody, Head, DTYPE_FEATURES, DTYPE_INDICES, DTYPE_SCORES

from boomer.algorithm.head_refinement import HeadRefinement, SingleLabelHeadRefinement
from boomer.algorithm.model import Theory
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

    h = None

    head: Head = None

    leq_conditions: Dict[int, float] = {}

    gr_conditions: Dict[int, float] = {}

    def reset(self):
        self.h = None
        self.head = None
        self.leq_conditions.clear()
        self.leq_conditions.clear()

    def apply_refinement(self, refinement: Refinement) -> 'RuleBuilder':
        """
        Applies a specific 'Refinement'.

        :param refinement:  The refinement to be applied
        :return:            The builder
        """
        self.h = refinement.h
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

    rule_builder = RuleBuilder()

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
        self.presorted_indices = None
        self.__validate()

        # Convert feature matrix into Fortran-contiguous array
        x = np.asfortranarray(x, dtype=DTYPE_FEATURES)

        # Convert binary ground truth labeling into expected confidence scores {-1, 1}
        expected_scores = np.asfortranarray(np.where(y > 0, y, -1), dtype=DTYPE_SCORES)

        # Induce default rule
        log.info('Learning rule 1 / %s (default rule)...', self.num_rules)
        default_rule = self.__induce_default_rule(expected_scores)

        # Initialize the confidence scores that are predicted by the default rule for each example and label
        predicted_scores = np.asfortranarray(np.tile(default_rule.head.scores, (expected_scores.shape[0], 1)))

        # Create initial theory
        theory = [default_rule]

        # TODO Induce more rules

        self.presorted_indices = None
        return theory

    def __validate(self):
        """
        Raises exceptions if the module is not configured properly.
        """

        if self.num_rules < 1:
            raise ValueError('Parameter \'num_rules\' must be at least 1, got {0}'.format(self.num_rules))
        if self.head_refinement is None:
            raise ValueError('Parameter \'head_refinement\' may not be None')

    def __induce_default_rule(self, expected_scores: np.ndarray) -> Rule:
        """
        Induces the default rule.

        :param expected_scores:     An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                    expected confidence scores according to the ground truth
        :return:                    The default rule
        """
        self.head_refinement.random_state = self.random_state
        head = self.head_refinement.find_default_head(expected_scores)
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
        self.rule_builder.reset()
        i = 1

        while True:
            refinement = self.__find_best_refinement(x, expected_scores, predicted_scores, presorted_indices, i)

            if refinement is not None and (self.rule_builder.h is None or refinement.h < self.rule_builder.h):
                self.rule_builder.apply_refinement(refinement)
                x = x[refinement.covered_indices]
                expected_scores = expected_scores[refinement.covered_indices]
                predicted_scores = predicted_scores[refinement.covered_indices]
                presorted_indices = None
                i += 1
            else:
                break

        return self.rule_builder.build()

    def __find_best_refinement(self, x: np.ndarray, expected_scores: np.ndarray, predicted_scores: np.ndarray,
                               presorted_indices: np.ndarray, iteration: int) -> Refinement:
        x, presorted_indices = self.__sub_sample_features(x, presorted_indices, iteration=iteration)
        # TODO: Do we really need to sort each time?
        sorted_indices = presorted_indices if presorted_indices is not None else GradientBoosting.presort_features(x)
        best_refinement = None

        for feature_index in range(0, get_num_features(sorted_indices)):
            indices = sorted_indices[:, feature_index]
            new_refinement = self.__find_best_condition(x, indices, expected_scores[indices], predicted_scores[indices],
                                                        feature_index)

            if best_refinement is None or (new_refinement is not None and new_refinement.h < best_refinement.h):
                best_refinement = new_refinement

        return best_refinement

    def __find_best_condition(self, x: np.ndarray, sorted_indices: np.ndarray, expected_scores: np.ndarray,
                              predicted_scores: np.ndarray, feature_index: int) -> Refinement:
        best_refinement = None
        previous_threshold = x[sorted_indices[0], feature_index]

        for r in range(1, get_num_examples(sorted_indices) - 1):
            threshold = x[sorted_indices[r], feature_index]

            # TODO Check if the second part of the if-condition is a good idea
            if previous_threshold != threshold and not np.array_equal(expected_scores[r - 1, :], expected_scores[r, :]):
                # LEQ
                head, h = self.head_refinement.find_head(expected_scores[:r, :], predicted_scores[:r, :])

                if best_refinement is None or h <= best_refinement.h:
                    best_refinement = Refinement(h=h, leq=True, threshold=GradientBoosting.__calculate_threshold(
                        previous_threshold, threshold), feature_index=feature_index, threshold_index=r,
                                                 head=head, covered_indices=sorted_indices[:r])

                # GR
                head, h = self.head_refinement.find_head(expected_scores[r:, :], predicted_scores[r:, :])

                if h < best_refinement.h:
                    best_refinement = Refinement(h=h, leq=False, threshold=GradientBoosting.__calculate_threshold(
                        previous_threshold, threshold), feature_index=feature_index, threshold_index=r,
                                                 head=head, covered_indices=sorted_indices[r:])

            previous_threshold = threshold

        return best_refinement

    def __sub_sample_features(self, x: np.ndarray, presorted_indices: np.ndarray,
                              iteration: int) -> (np.ndarray, np.ndarray):
        """
        Sub-samples the features.

        :param x:                   An array of dtype float, shape `(num_examples, num_features)`, representing the
                                    features of the training examples
        :param presorted_indices:   An array of dtype int, shape `(num_examples, num_features)`, representing the
                                    row-indices of the original examples at a certain position when sorting column-wise
                                    or None, if such an array is not available
        :param iteration:           The current iteration of the rule refinement process starting at 1
        :return:                    The
        """
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

    @staticmethod
    def __update_refinement(refinement: Refinement, h: float, threshold: float, feature_index: int, head: Head,
                            r: int, covered_indices: np.ndarray, leq: bool) -> Refinement:
        if refinement is None:
            return Refinement(h=h, leq=leq, threshold=threshold, feature_index=feature_index, threshold_index=r,
                              head=head, covered_indices=covered_indices)
        else:
            refinement.h = h
            refinement.leq = leq
            refinement.threshold = threshold
            refinement.feature_index = feature_index
            refinement.threshold_index = r
            refinement.head = head
            refinement.covered_indices = covered_indices
            return refinement
