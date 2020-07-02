from boomer.common._arrays cimport intp
from boomer.common.losses cimport RefinementSearch, Prediction
from boomer.common.head_refinement cimport HeadRefinement, HeadCandidate
from boomer.seco.lift_functions cimport LiftFunction


cdef class PartialHeadRefinement(HeadRefinement):

    # Attributes:

    cdef LiftFunction lift

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices,
                                 RefinementSearch refinement_search, bint uncovered, bint accumulated)

    cdef Prediction calculate_prediction(self, RefinementSearch refinement_search, bint uncovered, bint accumulated)
