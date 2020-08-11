from boomer.common._arrays cimport intp, float64
from boomer.common.statistics cimport AbstractRefinementSearch
from boomer.common.rule_evaluation cimport Prediction


cdef extern from "cpp/head_refinement.h" nogil:

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

    cdef HeadCandidate* find_head(self, HeadCandidate* best_head, HeadCandidate* recyclable_head,
                                  const intp* label_indices, AbstractRefinementSearch* refinement_search,
                                  bint uncovered, bint accumulated) nogil

    cdef Prediction* calculate_prediction(self, AbstractRefinementSearch* refinement_search, bint uncovered,
                                          bint accumulated) nogil


cdef class SingleLabelHeadRefinement(HeadRefinement):

    # Functions:

    cdef HeadCandidate* find_head(self, HeadCandidate* best_head, HeadCandidate* recyclable_head,
                                  const intp* label_indices, AbstractRefinementSearch* refinement_search,
                                  bint uncovered, bint accumulated) nogil

    cdef Prediction* calculate_prediction(self, AbstractRefinementSearch* refinement_search, bint uncovered,
                                          bint accumulated) nogil
