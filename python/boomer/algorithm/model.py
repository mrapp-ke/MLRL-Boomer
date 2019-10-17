#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides model classes, e.g., for representing rules.
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

    @staticmethod
    def match(body: 'Body', x: np.ndarray) -> np.ndarray:
        """
        Allows to check whether several examples are covered by a body, or not.

        :param body:    The body
        :param x:       An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                        examples to be matched
        :return:        An array of dtype bool, shape `(num_examples,)`, specifying for each example whether it is
                        covered by the body, or not
        """

        return np.all(np.less_equal(x[:, body.leq_features], body.leq_thresholds), axis=1) & np.all(
            np.greater(x[:, body.gr_features], body.gr_thresholds), axis=1)

    @staticmethod
    def filter_examples(body: 'Body', x: np.ndarray) -> np.ndarray:
        """
        Filters those examples that are covered by a body.

        :param body:    The body
        :param x:       An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                        examples to be matched
        :return:        An array of dtype float, shape `(num_covered, num_features)`, representing the features of the
                        covered examples
        """

        return x[Body.match(body, x)]

    @staticmethod
    def filter_labels(body: 'Body', x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Filters the label vectors of those examples that are covered by a body.

        :param body:    The body
        :param x:       An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                        examples to be matched
        :param y:       An array of dtype float, shape `(num_examples, num_labels)`, representing the labels of the
                        given examples
        :return:        An array of dtype float, shape `(num_covered, num_labels)`, representing the label vectors of
                        the covered examples
        """

        return y[Body.match(body, x)]


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


class Stats:
    """
    Stores useful information about a multi-label training data set, such as the number of examples, features, and
    labels.
    """

    def __init__(self, num_examples: int, num_features: int, num_labels: int):
        """
        :param num_examples:    The number of examples contained in the training data set
        :param num_features:    The number of features contained in the training data set
        :param num_labels:      The number of labels contained in the training data set
        """
        self.num_examples = num_examples
        self.num_features = num_features
        self.num_labels = num_labels

    @staticmethod
    def create_stats(x: np.ndarray, y: np.ndarray) -> 'Stats':
        """
        Creates 'Stats' storing information about a specific multi-label training data set.

        :param x:   An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                    training examples
        :param y:   An array of dtype float, shape `(num_examples, num_labels)`, representing the labels of the training
                    examples
        :return:    'Stats' storing information about the given data set
        """

        x_shape = np.shape(x)
        num_examples = x_shape[0]
        num_features = x_shape[1]
        num_labels = np.shape(y)[1]
        return Stats(num_examples=num_examples, num_features=num_features, num_labels=num_labels)
