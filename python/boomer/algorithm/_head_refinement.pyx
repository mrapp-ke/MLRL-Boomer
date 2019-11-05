# cython: boundscheck=False
# cython: wraparound=False
from boomer.algorithm._model import DTYPE_INDICES
import numpy as np


cdef class HeadCandidate:
    """
    Represent a candidate head.
    """

    def __cinit__(self, PartialHead head, float64 h):
        """
        :param head:    The partial head
        :param h:       The score that measures the quality of the head
        """
        self.head = head
        self.h = h


cdef class HeadRefinement:
    """
    A base class for all classes that allow to find optimal heads that minimize a certain loss functions for a given
    region of the instance space covered by a rule.
    """

    cdef FullHead find_default_head(self, float64[::1, :] expected_scores, DecomposableLoss loss):
        """
        Finds and returns the head of the default rule that minimizes a decomposable loss function with respect to the
        expected confidence scores according to the ground truth.

        :param expected_scores: An array of dtype float, shape `(num_examples, num_labels)`, representing the expected
                                confidence scores according to the ground truth
        :param loss:            A decomposable loss function
        :return:                The full head that has been found
        """
        cdef float64[::1, :] gradients = loss.calculate_initial_gradients(expected_scores)
        cdef float64[::1] scores = loss.calculate_optimal_scores(gradients)
        cdef FullHead head = FullHead(scores)
        return head

    cdef HeadCandidate find_head(self, float64[::1, :] expected_scores, float64[::1, :] predicted_scores,
                                 DecomposableLoss loss):
        """
        Finds and returns the head of a rule that minimizes a decomposable loss function with respect to the expected
        confidence scores for the examples covered by the rule according to the ground truth.

        :param expected_scores:     An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                    expected confidence scores for the examples covered by the rule according to the
                                    ground truth
        :param predicted_scores:    An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                    currently predicted confidence scores
        :param loss:                A decomposable loss function
        :return:                    The partial head that has been found, as well as a score that measures its quality
        """
        pass


cdef class SingleLabelHeadRefinement(HeadRefinement):
    """
    Allows to find single-label heads that minimize a certain loss function.
    """

    cdef HeadCandidate find_head(self, float64[::1, :] expected_scores, float64[::1, :] predicted_scores,
                                 DecomposableLoss loss):
        cdef float64[::1, :] gradients = loss.calculate_gradients(expected_scores, predicted_scores)
        cdef float64[::1] scores = loss.calculate_optimal_scores(gradients)
        cdef label_indices = np.linspace(0, scores.size, num=scores.size, endpoint=False, dtype=DTYPE_INDICES)
        cdef float64 h = loss.evaluate_predictions(scores, gradients)
        cdef PartialHead head = PartialHead(label_indices, scores)
        cdef HeadCandidate result = HeadCandidate(head, h)
        return result
