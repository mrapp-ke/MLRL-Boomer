from boomer.algorithm._arrays cimport intp, float64
from boomer.algorithm.lift_functions cimport LiftFunction
from boomer.algorithm.losses cimport Loss, Prediction


"""
A struct that stores a value of type float64 and a corresponding index that refers to the (original) position of the
value in an array.
"""
cdef struct IndexedValue:
    intp index
    float64 value

cdef class HeadCandidate:

    # Attributes:

    cdef readonly intp[::1] label_indices

    cdef readonly float64[::1] predicted_scores

    cdef readonly float64 quality_score


cdef class HeadRefinement:

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered,
                                 bint accumulated)

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered, bint accumulated)


cdef class FullHeadRefinement(HeadRefinement):

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered,
                                 bint accumulated)

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered, bint accumulated)


cdef class PartialHeadRefinement(HeadRefinement):

    cdef LiftFunction lift

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered)

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered)


cdef class SingleLabelHeadRefinement(HeadRefinement):

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered,
                                 bint accumulated)

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered, bint accumulated)
