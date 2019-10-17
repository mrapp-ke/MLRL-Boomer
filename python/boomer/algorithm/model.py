#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for representing the model learned by a classifier or ranker.
"""
from typing import List

import numpy as np

# Type alias for a theory, which is a list containing several rules
Theory = List['Rule']

# Type alias for the head of a rule. A head assigns a numerical score to each label.
Head = np.ndarray


class Body:
    """
    The body of a rule. A body is a conjunction of numerical conditions using <= and > operators.
    """

    def __init__(self, leq_features: np.ndarray, leq_thresholds: np.ndarray, gr_features: np.ndarray,
                 gr_thresholds: np.ndarray):
        """
        :param leq_features:    An array of dtype int, shape `(num_leq_conditions)`, representing the features of the
                                conditions that use the <= operator
        :param leq_thresholds:  An array of dtype float, shape `(num_leq_condition)`, representing the thresholds of the
                                conditions that use the <= operator
        :param gr_features:     An array of dtype int, shape `(num_gr_conditions)`, representing the features of the
                                conditions that use the > operator
        :param gr_thresholds:   An array of dtype float, shape `(num_gr_conditions)`, representing the thresholds of the
                                conditions that use the > operator
        """
        self.leq_features = leq_features
        self.leq_thresholds = leq_thresholds
        self.gr_features = gr_features
        self.gr_thresholds = gr_thresholds

    def match(self, x: np.ndarray) -> np.ndarray:
        """
        Allows to check whether several examples are covered by the body, or not.

        :param x:       An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                        examples to be matched
        :return:        An array of dtype bool, shape `(num_examples,)`, specifying for each example whether it is
                        covered by the body, or not
        """

        return np.all(np.less_equal(x[:, self.leq_features], self.leq_thresholds), axis=1) & np.all(
            np.greater(x[:, self.gr_features], self.gr_thresholds), axis=1)


class Rule:
    """
    A rule consisting of a body and head.
    """

    def __init__(self, body: Body, head: Head):
        """
        :param body:    The body of the rule
        :param head:    The head of the rule
        """
        self.body = body
        self.head = head
