# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np
from libc.math cimport pow
from boomer.algorithm._model import DTYPE_SCORES
ctypedef np.float64_t float64

cdef class Loss:
    """
    A base class for all loss functions.
    """


cdef class DecomposableLoss(Loss):
    """
    A base class for all decomposable loss functions.
    """

    cdef float64[::1, :] calculate_gradients(self, float64[::1, :] expected_scores, float64[::1, :] predicted_scores):
        """
        Calculates the gradient statistics, i.e., the first derivative of the loss function, given expected and
        predicted scores for individual examples and labels.

        :param expected_scores:     An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                    expected confidence scores according to the ground truth
        :param predicted_scores:    An array of dtype float, shape `(num_examples, num_labels`, representing the
                                    currently predicted confidence scores
        :return:                    An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                    gradient statistics for each example and label
        """
        pass

    cdef float64[::1] calculate_optimal_scores(self, float64[::1, :] gradients):
        """
        Calculates the optimal scores to be predicted for each label.

        :param gradients:   An array of dtype float, shape `(num_examples, num_labels)`, representing the gradient
                            statistics for individual examples and labels
        :return:            An array of dtype float, shape `(num_labels)', representing the optimal scores to be
                            predicted for each label
        """
        pass

    cdef float64 evaluate_predictions(self, float64[::1] scores, float64[::1, :] gradients):
        # TODO comment
        pass


cdef class SquaredErrorLoss(DecomposableLoss):
    """
    A multi-label variant of the squared error loss.
    """

    cdef float64[::1, :] calculate_gradients(self, float64[::1, :] expected_scores, float64[::1, :] predicted_scores):
        cdef Py_ssize_t num_rows = expected_scores.shape[0]
        cdef Py_ssize_t num_cols = expected_scores.shape[1]
        cdef float64[::1, :] gradients = np.empty((num_rows, num_cols), dtype=DTYPE_SCORES, order='F')
        cdef Py_ssize_t r, c

        for c in range(num_cols):
            for r in range(num_rows):
                gradients[r, c] = (2 * predicted_scores[r, c]) - (2 * expected_scores[r, c])

        return gradients

    cdef float64[::1] calculate_optimal_scores(self, float64[::1, :] gradients):
        cdef Py_ssize_t num_rows = gradients.shape[0]
        cdef Py_ssize_t num_cols = gradients.shape[1]
        cdef float64[::1] scores = np.empty((num_cols), dtype=DTYPE_SCORES)
        cdef Py_ssize_t r, c
        cdef float64 sum

        for c in range(num_cols):
            sum = 0

            for r in range(num_rows):
                sum += gradients[r, c]

            scores[c] = -sum / (2 * num_rows)

        return scores

    cdef float64 evaluate_predictions(self, float64[::1] scores, float64[::1, :] gradients):
        cdef Py_ssize_t num_rows = gradients.shape[0]
        cdef Py_ssize_t num_cols = gradients.shape[1]
        cdef float64 h = 0
        cdef Py_ssize_t r, c
        cdef float64 score

        for c in range(num_cols):
            score = scores[c]

            for r in range(num_rows):
                h += gradients[r, c] * score + pow(score, 2)

        return h
