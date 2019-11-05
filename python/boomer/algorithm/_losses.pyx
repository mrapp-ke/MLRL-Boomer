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

    cpdef float64[::1, :] calculate_initial_gradients(self, float64[::1, :] expected_scores):
        """
        Calculates the initial gradient statistics, i.e, the first derivative of the loss function, when always
        predicting 0, given expected scores for individual examples and labels.

        :param expected_scores: An array of dtype float, shape `(num_examples, num_labels)`, representing the expected
                                confidence scores according to the ground truth
        :return:                An array of dtype float, shape `(num_examples, num_labels)`, representing the initial
                                gradient statistics for each examples and label
        """
        pass

    cpdef float64[::1, :] calculate_gradients(self, float64[::1, :] expected_scores, float64[::1, :] predicted_scores):
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

    cpdef float64[::1] calculate_optimal_scores(self, float64[::1, :] gradients):
        """
        Calculates the optimal scores to be predicted for each label.

        :param gradients:   An array of dtype float, shape `(num_examples, num_labels)`, representing the gradient
                            statistics for individual examples and labels
        :return:            An array of dtype float, shape `(num_labels)', representing the optimal scores to be
                            predicted for each label
        """
        pass

    cpdef float64 evaluate_predictions(self, float64[::1] scores, float64[::1, :] gradients):
        """
        Calculates a single score that measures the quality of predictions.

        :param scores:      An array of dtype float, shape `(num_labels)`, representing the scores predicted for each
                            label
        :param gradients:   An array of dtype float, shape `(num_examples, num_labels)`, representing the gradient
                            statistics for individual examples and labels
        :return:            A scalar of dtype float, representing the calculated score
        """
        pass


cdef class SquaredErrorLoss(DecomposableLoss):
    """
    A multi-label variant of the squared error loss.
    """

    cpdef float64[::1, :] calculate_initial_gradients(self, float64[::1, :] expected_scores):
        cdef Py_ssize_t num_rows = expected_scores.shape[0]
        cdef Py_ssize_t num_cols = expected_scores.shape[1]
        cdef float64[::1, :] gradients = np.empty((num_rows, num_cols), dtype=DTYPE_SCORES, order='F')
        cdef Py_ssize_t r, c

        for c in range(num_cols):
            for r in range(num_rows):
                gradients[r, c] = -2 * expected_scores[r, c]

        return gradients


    cpdef float64[::1, :] calculate_gradients(self, float64[::1, :] expected_scores, float64[::1, :] predicted_scores):
        cdef Py_ssize_t num_rows = expected_scores.shape[0]
        cdef Py_ssize_t num_cols = expected_scores.shape[1]
        cdef float64[::1, :] gradients = np.empty((num_rows, num_cols), dtype=DTYPE_SCORES, order='F')
        cdef Py_ssize_t r, c

        for c in range(num_cols):
            for r in range(num_rows):
                gradients[r, c] = (2 * predicted_scores[r, c]) - (2 * expected_scores[r, c])

        return gradients

    cpdef float64[::1] calculate_optimal_scores(self, float64[::1, :] gradients):
        cdef Py_ssize_t num_rows = gradients.shape[0]
        cdef Py_ssize_t num_cols = gradients.shape[1]
        cdef float64[::1] scores = np.empty((num_cols), dtype=DTYPE_SCORES)
        cdef Py_ssize_t r, c
        cdef float64 sum_of_gradients
        cdef float64 sum_of_hessians = 2 * num_rows 

        for c in range(num_cols):
            sum_of_gradients = 0

            for r in range(num_rows):
                sum_of_gradients += gradients[r, c]

            scores[c] = -sum_of_gradients / sum_of_hessians

        return scores

    cpdef float64 evaluate_predictions(self, float64[::1] scores, float64[::1, :] gradients):
        cdef Py_ssize_t num_rows = gradients.shape[0]
        cdef Py_ssize_t num_cols = gradients.shape[1]
        cdef float64 h = 0
        cdef Py_ssize_t r, c
        cdef float64 score, score_pow

        for c in range(num_cols):
            score = scores[c]
            score_pow = pow(score, 2)

            for r in range(num_rows):
                h += (gradients[r, c] * score) + score_pow

        return h
