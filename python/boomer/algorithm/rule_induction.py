#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for inducing classification rules.
"""
import logging as log
from abc import abstractmethod

import numpy as np
from boomer.algorithm._head_refinement import HeadRefinement
from boomer.algorithm._losses import Loss
from boomer.algorithm._pruning import Pruning
from boomer.algorithm._rule_induction import induce_default_rule, induce_rule
from boomer.algorithm._sub_sampling import InstanceSubSampling, FeatureSubSampling

from boomer.algorithm.model import Theory, DTYPE_INTP, DTYPE_UINT8, DTYPE_FLOAT32
from boomer.algorithm.stats import Stats
from boomer.learners import Module


class RuleInduction(Module):
    """
    A module that allows to induce a `Theory`, consisting of several classification rules.
    """

    @abstractmethod
    def induce_rules(self, stats: Stats, x: np.ndarray, y: np.ndarray, theory: Theory) -> Theory:
        """
        Creates and returns a 'Theory' that contains several candidate rules.

        :param stats:   Statistics about the training data set
        :param x:       An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                        training examples
        :param y:       An array of dtype float, shape `(num_examples, num_labels)`, representing the labels of the
                        training examples
        :param theory:  An existing theory, the induced classification rules should be added to, or None if a new theory
                        should be created
        :return:        A 'Theory' that contains the induced classification rules
        """
        pass


class GradientBoosting(RuleInduction):
    """
    Implements the induction of (multi-label) classification rules using gradient boosting.
    """

    def __init__(self, num_rules, head_refinement: HeadRefinement, loss: Loss,
                 instance_sub_sampling: InstanceSubSampling, feature_sub_sampling: FeatureSubSampling, pruning: Pruning,
                 shrinkage: float):
        """
        :param num_rules:               The number of rules to be induced (including the default rule)
        :param head_refinement:         The strategy that is used to find the heads of rules
        :param loss:                    The loss function to be minimized
        :param instance_sub_sampling:   The strategy that is used for sub-sampling the training examples each time a new
                                        classification rule is learned
        :param feature_sub_sampling:    The strategy that is used for sub-sampling the features each time a
                                        classification rule is refined
        :param pruning:                 The strategy that is used for pruning rules
        :param shrinkage:               The shrinkage parameter that should be applied to the predictions of newly
                                        induced rules to reduce their effect on the entire model. Must be in (0, 1]
        """
        self.num_rules = num_rules
        self.head_refinement = head_refinement
        self.loss = loss
        self.instance_sub_sampling = instance_sub_sampling
        self.feature_sub_sampling = feature_sub_sampling
        self.pruning = pruning
        self.shrinkage = shrinkage

    def induce_rules(self, stats: Stats, x: np.ndarray, y: np.ndarray, theory: Theory) -> Theory:
        self.__validate()
        num_rules = self.num_rules
        random_state = self.random_state
        head_refinement = self.head_refinement
        loss = self.loss
        instance_sub_sampling = self.instance_sub_sampling
        feature_sub_sampling = self.feature_sub_sampling
        pruning = self.pruning
        shrinkage = self.shrinkage

        # Convert feature and label matrices into Fortran-contiguous arrays
        x = np.asfortranarray(x, dtype=DTYPE_FLOAT32)
        y = np.asfortranarray(y, dtype=DTYPE_UINT8)

        # Sort feature matrix once
        x_sorted_indices = np.asfortranarray(np.argsort(x, axis=0), dtype=DTYPE_INTP)

        # Create a new theory, if necessary
        if theory is None:
            theory = []

        # Induce default rule, if necessary
        if len(theory) == 0:
            log.info('Learning rule 1 / %s (default rule)...', num_rules)
            default_rule = induce_default_rule(y, loss)
            theory.append(default_rule)

        for i in range(len(theory) + 1, num_rules + 1):
            log.info('Learning rule %s / %s...', i, num_rules)

            # Induce a new rule
            rule = induce_rule(x, x_sorted_indices, head_refinement, loss, instance_sub_sampling, feature_sub_sampling,
                               pruning, shrinkage, random_state)

            # Add new rule to theory
            theory.append(rule)

            # Alter random state for inducing the next rule
            random_state += 1

        return theory

    def __validate(self):
        """
        Raises exceptions if the module is not configured properly.
        """

        if self.num_rules < 1:
            raise ValueError('Parameter \'num_rules\' must be at least 1, got {0}'.format(self.num_rules))
        if self.head_refinement is None:
            raise ValueError('Parameter \'head_refinement\' may not be None')
        if self.shrinkage <= 0:
            raise ValueError('Parameter \'shrinkage\' must be greater than 0, got {0}'.format(self.shrinkage))
        if self.shrinkage > 1:
            raise ValueError('Parameter \'shrinkage\' must be at maximum 1, got {0}'.format(self.shrinkage))
