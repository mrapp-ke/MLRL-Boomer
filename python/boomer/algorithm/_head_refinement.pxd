from boomer.algorithm._model cimport float64, FullHead, PartialHead
from boomer.algorithm._losses cimport DecomposableLoss

cdef class HeadCandidate:

    cdef readonly PartialHead head

    cdef readonly float64 h


cdef class HeadRefinement:

    cdef FullHead find_default_head(self, float64[::1, :] expected_scores, DecomposableLoss loss)

    cdef HeadCandidate find_head(self, float64[::1, :] expected_scores, float64[::1, :] predicted_scores,
                                 DecomposableLoss loss)


cdef class SingleLabelHeadRefinement(HeadRefinement):

    cdef FullHead find_default_head(self, float64[::1, :] expected_scores, DecomposableLoss loss)

    cdef HeadCandidate find_head(self, float64[::1, :] expected_scores, float64[::1, :] predicted_scores,
                                 DecomposableLoss loss)
