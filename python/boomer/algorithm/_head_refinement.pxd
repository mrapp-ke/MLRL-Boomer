from boomer.algorithm._model cimport intp, float64
from boomer.algorithm._losses cimport Loss


cdef class HeadCandidate:

    # Attributes:

    cdef readonly intp[::1] label_indices

    cdef readonly float64[::1] predicted_scores

    cdef readonly float64 quality_score


cdef class HeadRefinement:

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate current_head, Loss loss, bint covered)


cdef class SingleLabelHeadRefinement(HeadRefinement):

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate current_head, Loss loss, bint covered)
