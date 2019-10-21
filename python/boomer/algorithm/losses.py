#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that represent (surrogate) loss functions for use in gradient boosting.
"""
from abc import ABC, abstractmethod

import numpy as np

from boomer.algorithm.stats import get_num_examples


class Loss(ABC):
    """
    A base class for all loss functions.
    """


class DecomposableLoss(Loss):
    """
    A base class for all decomposable loss functions.
    """

    @abstractmethod
    def derive_scores(self, expected_scores: np.ndarray, predicted_scores: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Derives the optimal scores to be predicted for each label.

        :param expected_scores:     An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                    expected confidence scores according to the ground truth
        :param predicted_scores:    An array of dtype float, shape `(num_examples, num_labels`, representing the
                                    currently predicted confidence scores
        :return:                    An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                    optimal scores to be predicted by a rule for each label, as well as an array of type
                                    float, shape `(num_examples, num_labels)`, representing the gradients for each
                                    example and label
        """
        pass

    def evaluate_predictions(self, scores: np.ndarray, first_derivative: np.ndarray) -> float:
        # TODO comment
        pass


class SquaredErrorLoss(DecomposableLoss):
    """
    A multi-label variant of the squared error loss.
    """

    def derive_scores(self, expected_scores: np.ndarray, predicted_scores: np.ndarray) -> (np.ndarray, np.ndarray):
        first_derivative = (2 * predicted_scores) - (2 * expected_scores)
        scores = -np.sum(first_derivative, axis=0) / (get_num_examples(expected_scores) * 2)
        return scores, first_derivative

    def evaluate_predictions(self, scores: np.ndarray, first_derivative: np.ndarray) -> float:
        # gradient * score + 1/2 * hessian * score^2 = gradient * score + score^2
        return np.sum((first_derivative * scores) + np.square(scores)).item()
