from boomer.common._arrays cimport intp
from boomer.common._predictions cimport PredictionCandidate
from boomer.common.statistics cimport AbstractRefinementSearch
from boomer.common.head_refinement cimport HeadRefinement


cdef class FullHeadRefinement(HeadRefinement):

    # Functions:

    cdef PredictionCandidate* find_head(self, PredictionCandidate* best_head, PredictionCandidate* recyclable_head,
                                        const intp* label_indices, AbstractRefinementSearch* refinement_search,
                                        bint uncovered, bint accumulated) nogil

    cdef PredictionCandidate* calculate_prediction(self, AbstractRefinementSearch* refinement_search, bint uncovered,
                                                   bint accumulated) nogil
