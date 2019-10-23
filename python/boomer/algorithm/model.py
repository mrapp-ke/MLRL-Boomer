#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for representing the model learned by a classifier or ranker.
"""
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from boomer.algorithm.stats import get_num_examples

# The dtype used for indices
DTYPE_INDICES = np.int32

# The dtype used for numerical scores
DTYPE_SCORES = np.float64

# The dtype used for feature values
DTYPE_FEATURES = np.float32

# Type alias for a theory, which is a list containing several rules
Theory = List['Rule']


class Body(ABC):
    """
    A base class for the body of a rule.
    """

    @abstractmethod
    def match(self, x: np.ndarray) -> np.ndarray:
        """
        Allows to check whether several examples are covered by the body, or not.

        :param x:       An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                        examples to be matched
        :return:        An array of dtype bool, shape `(num_examples,)`, specifying for each example whether it is
                        covered by the body, or not
        """
        pass


class EmptyBody(Body):
    """
    An empty body that matches all examples.
    """

    def match(self, x: np.ndarray) -> np.ndarray:
        return np.full((get_num_examples(x)), True)


class ConjunctiveBody(Body):
    """
    A body that given as a conjunction of numerical conditions using <= and > operators.
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
        return np.all(np.less_equal(x[:, self.leq_features], self.leq_thresholds), axis=1) & np.all(
            np.greater(x[:, self.gr_features], self.gr_thresholds), axis=1)


class Head(ABC):
    """
    A base class for the head of a rule.
    """

    @abstractmethod
    def predict(self, predictions: np.ndarray):
        """
        Applies the head's prediction to a given matrix of predictions.

        :param predictions:     An array of dtype float, shape `(num_examples, num_labels)`, representing the scores
                                predicted for the corresponding examples
        """
        pass


class FullHead(Head):
    """
    A full head that assigns a numerical score to each label.
    """

    def __init__(self, scores: np.ndarray):
        """
        :param scores:  An array of dtype float, shape `(num_labels)`, representing the scores that are predicted by the
                        rule for each label
        """
        self.scores = scores

    def predict(self, predictions: np.ndarray):
        predictions += self.scores


class PartialHead(Head):
    """
    A partial head that assigns a numerical score to one or several labels.
    """

    def __init__(self, scores: np.ndarray, labels: np.ndarray):
        """
        :param labels:  An array of dtype int, shape `(num_predicted_labels)`, representing the indices of the labels
                        for which the rule predicts
        :param scores:  An array of dtype float, shape `(num_predicted_labels)`, representing the scores that are
                        predicted by the rule
                """
        self.scores = scores
        self.labels = labels

    def predict(self, predictions: np.ndarray):
        predictions[:, self.labels] += self.scores


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

    def predict(self, x: np.ndarray, predictions: np.ndarray):
        """
        Applies the rule's prediction to all examples it covers.

        :param x:               An array of dtype float, shape `(num_examples, num_features)`, representing the features
                                of the examples to predict for
        :param predictions:     An array of dtype float, shape `(num_examples, num_labels)`, representing the scores
                                predicted for the given examples
        """
        self.head.predict(predictions[self.body.match(x), :])


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
