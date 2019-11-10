# cython: boundscheck=False
# cython: wraparound=False
from cython.view cimport array as cvarray
from boomer.algorithm._model cimport intp


cdef class HeadCandidate:
    """
    Represents a candidate head.
    """

    def __cinit__(self, PartialHead head, float64 quality_score):
        """
        :param head:            The partial head
        :param quality_score:   A score that measures the quality of the head
        """
        self.head = head
        self.quality_score = quality_score


cdef class HeadRefinement:
    """
    A base class for all classes that allow to find optimal heads that minimize a certain loss functions for a given
    region of the instance space covered by a rule.
    """

    cdef HeadCandidate find_head(self, PartialHead current_head, Loss loss, bint covered):
        """
        Finds and returns the head of a rule that minimizes a loss function with respect to the current gradient
        statistics.

        :param current_head:    The current head of the rule or None, if no head has been found yet
        :param loss:            The loss function to be minimized
        :param covered:         1, if the rule for which the head should be found covers the examples that have been
                                provided to the loss function or 0, if the rule covers all other examples
        :return:                A 'HeadCandidate' consisting of the partial head that has been found, as well as its
                                quality score
        """
        pass


cdef class SingleLabelHeadRefinement(HeadRefinement):
    """
    Allows to find single-label heads that minimize a certain loss function.
    """

    cdef HeadCandidate find_head(self, PartialHead current_head, Loss loss, bint covered):
        cdef float64[::1] scores = loss.calculate_scores(covered)
        cdef float64[::1] quality_scores = loss.calculate_quality_scores(covered)
        cdef intp best_c = 0
        cdef float64 best_quality_score = quality_scores[best_c]
        cdef intp[::1] label_indices = cvarray(shape=(1,), itemsize=sizeof(intp), format='i', mode='c')
        cdef float64[::1] predicted_scores = cvarray(shape=(1,), itemsize=sizeof(float64), format='d', mode='c')
        cdef float64 quality_score
        cdef intp num_labels, c

        if current_head is None:
            num_labels = quality_scores.shape[0]

            for c in range(1, num_labels):
                quality_score = quality_scores[c]

                if quality_score < best_quality_score:
                    best_quality_score = quality_score
                    best_c = c

            label_indices[0] = best_c
        else:
            label_indices[0] = current_head.label_indices[0]

        predicted_scores[0] = scores[best_c]
        cdef PartialHead best_head = PartialHead(label_indices, predicted_scores)
        cdef HeadCandidate candidate = HeadCandidate(best_head, best_quality_score)
        return candidate
