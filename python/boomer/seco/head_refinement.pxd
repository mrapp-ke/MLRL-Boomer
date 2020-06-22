from boomer.common._arrays cimport intp, float64
from boomer.common.losses cimport Loss, Prediction
from boomer.common.head_refinement cimport HeadRefinement, HeadCandidate
from boomer.seco.lift_functions cimport LiftFunction


"""
A struct that stores a value of type float64 and a corresponding index that refers to the (original) position of the
value in an array.
"""
cdef struct IndexedValue:
    intp index
    float64 value


cdef class PartialHeadRefinement(HeadRefinement):

    # Attributes:

    cdef LiftFunction lift

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered,
                                 bint accumulated)

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered, bint accumulated)
