from boomer.algorithm._model cimport float64, PartialHead
from boomer.algorithm._losses cimport Loss


cdef class HeadCandidate:

    # Attributes:

    cdef readonly PartialHead head

    cdef readonly float64 quality_score


cdef class HeadRefinement:

    # Functions:

    cdef HeadCandidate find_head(self, PartialHead current_head, Loss loss, bint covered)


cdef class SingleLabelHeadRefinement(HeadRefinement):

    # Functions:

    cdef HeadCandidate find_head(self, PartialHead current_head, Loss loss, bint covered)
