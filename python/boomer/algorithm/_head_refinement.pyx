# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement strategies for finding the heads of rules.
"""
from cython.view cimport array as cvarray
from boomer.algorithm._utils cimport get_label_index


cdef class HeadCandidate:
    """
    Stores information about a potential head of a rule.
    """

    def __cinit__(self, intp[::1] label_indices, float64[::1] predicted_scores, float64 quality_score):
        """
        :param label_indices:       An array of dtype int, shape `(num_predicted_labels)`, representing the indices of
                                    the labels for which the head predicts or None, if the head predicts for all labels
        :param predicted_scores:    An array of dtype float, shape `(num_predicted_labels)`, representing the scores
                                    that are predicted by the head. The predicted scores correspond to the indices in
                                    the array `label_indices`.  If `label_indices` is None, the scores correspond to all
                                    labels in the training data
        :param quality_score:       A score that measures the quality of the head
        """
        self.label_indices = label_indices
        self.predicted_scores = predicted_scores
        self.quality_score = quality_score


cdef class HeadRefinement:
    """
    A base class for all classes that allow to find the best single- or multi-label head for a rule.
    """

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices,
                                 float64[::1, :] predicted_and_quality_scores, intp row_index):
        """
        Finds and returns the best head for a rule, given the predicted scores and quality scores for each label.

        :param best_head:                       The `HeadCandidate` that corresponds to the best rule known so far (as
                                                found in the previous or current refinement iteration) or None, if no
                                                such rule is available yet. The new head must be better than this one,
                                                otherwise it is discarded. If the new head is better, this
                                                `HeadCandidate` will be modified accordingly instead of creating a new
                                                instance to avoid unnecessary memory allocations
        :param label_indices:                   An array of dtype int, shape `(num_labels)`, representing the indices of
                                                the labels for which the head may predict or None, if the head may
                                                predict for all labels
        :param predicted_and_quality_scores:    An array of dtype float, shape `(num_rules * 2, num_labels)`, where the
                                                i-th row (starting at 0) represents the optimal scores to be predicted
                                                by a rule for the individual labels and the i+1-th row represents the
                                                corresponding quality scores
        :param row_index:                       The index of the row in 'predicted_and_quality_scores' that contains the
                                                optimal predictions of the rule for which the best head should be found
        :return:                                A 'HeadCandidate' that stores information about the head that has been
                                                found, if the head is better than `best_head`, None otherwise
        """
        pass


cdef class FullHeadRefinement(HeadRefinement):
    """
    Allows to find the best multi-label head that predicts for all labels.
    """

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices,
                                 float64[::1, :] predicted_and_quality_scores, intp row_index):
        cdef intp quality_score_index = row_index + 1
        cdef float64 quality_score = 0
        cdef intp num_labels = predicted_and_quality_scores.shape[1]
        cdef float64[::1] predicted_scores
        cdef HeadCandidate candidate
        cdef intp c

        # Calculate the quality score of the new head by summing up the quality scores for all labels...
        for c in range(num_labels):
            quality_score += predicted_and_quality_scores[quality_score_index, c]

        if best_head is None:
            # Create a new `HeadCandidate` and return it...
            predicted_scores = cvarray(shape=(num_labels,), itemsize=sizeof(float64), format='d', mode='c')

            for c in range(num_labels):
                predicted_scores[c] = predicted_and_quality_scores[row_index, c]

            candidate = HeadCandidate(None, predicted_scores, quality_score)
            return candidate
        else:
            # The quality score must be better than that of `best_head`...
            if quality_score < best_head.quality_score:
                # Modify the `best_head` and return it...
                for c in range(num_labels):
                    best_head.predicted_scores[c] = predicted_and_quality_scores[row_index, c]

                best_head.quality_score = quality_score
                return best_head

        # Return None, as the quality score of the found head is worse than that of `best_head`...
        return None


cdef class SingleLabelHeadRefinement(HeadRefinement):
    """
    Allows to find the best single-label head that predicts for a single label.
    """

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices,
                                 float64[::1, :] predicted_and_quality_scores, intp row_index):
        cdef intp best_c = 0
        cdef intp quality_score_index = row_index + 1
        cdef intp num_labels = predicted_and_quality_scores.shape[1]
        cdef float64 best_quality_score = predicted_and_quality_scores[quality_score_index, best_c]
        cdef intp[::1] predicted_label_indices
        cdef float64[::1] predicted_scores
        cdef HeadCandidate candidate
        cdef float64 quality_score
        cdef intp c

        # Find the best single-label head...
        for c in range(1, num_labels):
            quality_score = predicted_and_quality_scores[quality_score_index, c]

            if quality_score < best_quality_score:
                best_quality_score = quality_score
                best_c = c

        if best_head is None:
            # Create a new `HeadCandidate` and return it...
            predicted_label_indices = cvarray(shape=(1,), itemsize=sizeof(intp), format='l', mode='c')
            predicted_label_indices[0] = get_label_index(best_c, label_indices)
            predicted_scores = cvarray(shape=(1,), itemsize=sizeof(float64), format='d', mode='c')
            predicted_scores[0] = predicted_and_quality_scores[row_index, best_c]
            candidate = HeadCandidate(predicted_label_indices, predicted_scores, best_quality_score)
            return candidate
        else:
            # The quality score must be better than that of `best_head`...
            if best_quality_score < best_head.quality_score:
                best_head.label_indices[0] = get_label_index(best_c, label_indices)
                best_head.predicted_scores[0] = predicted_and_quality_scores[row_index, best_c]
                best_head.quality_score = best_quality_score
                return best_head

        # Return None, as the quality_score of the found head is worse than that of `best_head`...
        return None
