from boomer.algorithm._model cimport intp, float64


cdef class HeadCandidate:

    # Attributes:

    cdef readonly intp[::1] label_indices

    cdef readonly float64[::1] predicted_scores

    cdef readonly float64 quality_score


cdef class HeadRefinement:

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate best_head, HeadCandidate best_candidate,
                                 float64[::1, :] predicted_and_quality_scores, intp row_index)


cdef class SingleLabelHeadRefinement(HeadRefinement):

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate best_head, HeadCandidate best_candidate,
                                 float64[::1, :] predicted_and_quality_scores, intp row_index)
