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

    def first_derivative(self, ground_truth: np.ndarray, prediction: np.ndarray, index: int) -> float:
        """
        Calculates the first partial derivative of the loss with respect to the prediction at a specific index.

        :param ground_truth:    An array of dtype float, shape `(num_rows, num_cols)`, representing the expected
                                confidence scores according to the ground truth labeling
        :param prediction:     An array of dtype float, shape `(num_rows, num_cols)`, representing the predicted
                                confidence scores
        :param index            The index of the prediction
        :return:                The first partial derivative of the loss with respect to the given prediction
        """
        pass

    def second_derivative(self, ground_truth: np.ndarray, prediction: np.ndarray, index1: int, index2: int) -> float:
        """
        Calculates the second partial derivative of the loss with respect to the predictions at a specific indices.

        :param ground_truth:    An array of dtype float, shape `(num_rows, num_cols)`, representing the expected
                                confidence scores according to the ground truth labeling
        :param prediction:     An array of dtype float, shape `(num_rows, num_cols)`, representing the predicted
                                confidence scores
        :param index1           The index of the first prediction
        :param index2           The index of the second prediction
        :return:                The second partial derivative of the loss with respect to the given prediction
        """
        pass

    def is_decomposable(self) -> bool:
        """
        Returns whether the loss is element-wise decomposable or not.

        :return: 'True', if the loss is decomposable, 'False' otherwise
        """
        pass


class SquaredErrorLoss(Loss):
    """
    A multi-label variant of the squared error loss.
    """

    def first_derivative(self, ground_truth: np.ndarray, prediction: np.ndarray, index: int) -> float:
        return (2 * prediction[index]) - (2 * ground_truth[index])

    def second_derivative(self, ground_truth: np.ndarray, prediction: np.ndarray, index1: int, index2: int) -> float:
        return 2

    def is_decomposable(self) -> bool:
        return True
