from boomer.common._arrays cimport intp
from boomer.common.losses cimport PredictionSearch, Prediction
from boomer.common.head_refinement cimport HeadRefinement, HeadCandidate
from boomer.seco.lift_functions cimport LiftFunction


cdef class PartialHeadRefinement(HeadRefinement):

    # Attributes:

    cdef LiftFunction lift

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices,
                                 PredictionSearch prediction_search, bint uncovered, bint accumulated)

    cdef Prediction calculate_prediction(self, PredictionSearch prediction_search, bint uncovered, bint accumulated)
