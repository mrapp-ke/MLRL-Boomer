from boomer.common._arrays cimport intp, float64
from boomer.common.losses cimport RefinementSearch, Prediction


cdef extern from "cpp/head_refinement.h" namespace "head_refinement":

    cdef cppclass HeadCandidate:

        # Constructors:

        HeadCandidate(intp numPredictions, intp* labelIndices, float64* predictedScores, float64 qualityScore) except +

        # Attributes:

        intp numPredictions_

        intp* labelIndices_

        float64* predictedScores_

        float64 qualityScore_


cdef class HeadRefinement:

    # Functions:

    cdef HeadCandidate* find_head(self, HeadCandidate* best_head, intp[::1] label_indices,
                                  RefinementSearch refinement_search, bint uncovered, bint accumulated)

    cdef Prediction* calculate_prediction(self, RefinementSearch refinement_search, bint uncovered, bint accumulated)


cdef class SingleLabelHeadRefinement(HeadRefinement):

    # Functions:

    cdef HeadCandidate* find_head(self, HeadCandidate* best_head, intp[::1] label_indices,
                                  RefinementSearch refinement_search, bint uncovered, bint accumulated)

    cdef Prediction* calculate_prediction(self, RefinementSearch refinement_search, bint uncovered, bint accumulated)
