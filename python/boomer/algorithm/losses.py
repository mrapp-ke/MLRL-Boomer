#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that represent (surrogate) loss functions for use in gradient boosting.
"""
import numpy as np


class Loss:
    """
    A base class for all loss functions.
    """


class DecomposableLoss(Loss):
    """
    A base class for all decomposable loss functions.
    """

    def derive_scores(self, expected_scores: np.ndarray, predicted_scores: np.ndarray) -> np.ndarray:
        """
        Derives the optimal scores to be predicted for each label.

        :param expected_scores:     An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                    expected confidence scores according to the ground truth
        :param predicted_scores:    An array of dtype float, shape `(num_examples, num_labels`, representing the
                                    currently predicted confidence scores
        :return:                    An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                    optimal scores to be predicted by a rule for each label
        """
        pass


class SquaredErrorLoss(DecomposableLoss):
    """
    A multi-label variant of the squared error loss.
    """

    def derive_scores(self, expected_scores: np.ndarray, predicted_scores: np.ndarray) -> np.ndarray:
        return np.sum(-((2 * predicted_scores) - (2 * expected_scores)) / 2, axis=0)
