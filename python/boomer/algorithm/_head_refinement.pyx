# cython: boundscheck=False
# cython: wraparound=False
from cython.view cimport array as cvarray
from boomer.algorithm._model cimport intp


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

    cdef HeadCandidate find_head(self, HeadCandidate best_head, HeadCandidate best_candidate,
                                 float64[::1, :] predicted_and_quality_scores, intp row_index):
        """
        Finds and returns the best head for a rule, given the predicted scores and quality scores for each label.

        :param best_head:                       The `HeadCandidate` that corresponds to the best rule known so far (as
                                                found in the current refinement iteration or the previous one) or None,
                                                if no such rule is available yet. The new head must be better than
                                                `best_head`, otherwise it is discarded. If the new head is better, this
                                                `HeadCandidate` will be modified accordingly instead of creating a new
                                                instance to avoid unnecessary memory allocations
        :param best_candidate:                  The `HeadCandidate` that corresponds to the best refinement candidate
                                                known so far (as found in the current refinement iteration; may be
                                                worse and must not be better than the rule with `best_head`) or  None,
                                                if no such refinement candidate is available yet. As the new head must
                                                be better than `best_candidate` (and `best_head`), this allows to
                                                discard unpromising candidates
        :param predicted_and_quality_scores:    An array of dtype float, shape `(num_rules * 2, num_labels)`, where the
                                                i-th row (starting at 0) represents the optimal scores to be predicted
                                                by a rule for the individual labels and the i+1-th row represents the
                                                corresponding quality scores
        :param row_index:                       The index of the row in 'predicted_and_quality_scores' that contains the
                                                optimal predictions of the rule for which the best head should be found
        :return:                                A 'HeadCandidate' that stores information about the head that has been
                                                found, if the head is better than `best_head` and `best_candidate`, None
                                                otherwise
        """
        pass


cdef class FullHeadRefinement(HeadRefinement):
    """
    Allows to find multi-label heads that predict for all labels.
    """

    cdef HeadCandidate find_head(self, HeadCandidate best_head, HeadCandidate best_candidate,
                                 float64[::1, :] predicted_and_quality_scores, intp row_index):
        cdef intp quality_score_index = row_index + 1
        cdef float64 quality_score = 0
        cdef intp num_labels = predicted_and_quality_scores.shape[1]
        cdef intp[::1] label_indices
        cdef float64[::1] predicted_scores
        cdef HeadCandidate candidate
        cdef intp c

        for c in range(num_labels):
            quality_score += predicted_and_quality_scores[quality_score_index, c]

        if best_head is None:
            if best_candidate is None or quality_score < best_candidate.quality_score:
                label_indices = cvarray(shape=(num_labels,), itemsize=sizeof(intp), format='l', mode='c')
                predicted_scores = cvarray(shape=(num_labels,), itemsize=sizeof(float64), format='d', mode='c')

                for c in range(num_labels):
                    label_indices[c] = c
                    predicted_scores[c] = predicted_and_quality_scores[row_index, c]

                candidate = HeadCandidate(label_indices, predicted_scores, quality_score)
                return candidate
        elif (best_candidate is not None and quality_score < best_candidate.quality_score) or quality_score < best_head.quality_score:
            for c in range(num_labels):
                best_head.predicted_scores[c] = predicted_and_quality_scores[row_index, c]

            best_head.quality_score = quality_score
            return best_head

        return None


cdef class SingleLabelHeadRefinement(HeadRefinement):
    """
    Allows to find single-label heads that predict for a single label.
    """

    cdef HeadCandidate find_head(self, HeadCandidate best_head, HeadCandidate best_candidate,
                                 float64[::1, :] predicted_and_quality_scores, intp row_index):
        cdef intp best_c = 0
        cdef intp quality_score_index = row_index + 1
        cdef float64 best_quality_score = predicted_and_quality_scores[quality_score_index, best_c]
        cdef intp[::1] label_indices
        cdef float64[::1] predicted_scores
        cdef HeadCandidate candidate
        cdef float64 quality_score
        cdef intp num_labels, c

        if best_head is None:
            num_labels = predicted_and_quality_scores.shape[1]

            for c in range(1, num_labels):
                quality_score = predicted_and_quality_scores[quality_score_index, c]

                if quality_score < best_quality_score:
                    best_quality_score = quality_score
                    best_c = c

            if best_candidate is None or best_quality_score < best_candidate.quality_score:
                label_indices = cvarray(shape=(1,), itemsize=sizeof(intp), format='l', mode='c')
                label_indices[0] = best_c
                predicted_scores = cvarray(shape=(1,), itemsize=sizeof(float64), format='d', mode='c')
                predicted_scores[0] = predicted_and_quality_scores[row_index, best_c]
                candidate = HeadCandidate(label_indices, predicted_scores, best_quality_score)
                return candidate
        elif (best_candidate is not None and best_quality_score < best_candidate.quality_score) or best_quality_score < best_head.quality_score:
            best_head.predicted_scores[0] = predicted_and_quality_scores[row_index, best_c]
            best_head.quality_score = best_quality_score
            return best_head

        return None
