from boomer.common._arrays cimport intp
from boomer.common.statistics cimport RefinementSearch
from boomer.common.rule_evaluation cimport Prediction
from boomer.common.head_refinement cimport HeadRefinement, HeadCandidate


cdef class FullHeadRefinement(HeadRefinement):

    # Functions:

    cdef HeadCandidate* find_head(self, HeadCandidate* best_head, HeadCandidate* recyclable_head,
                                  intp[::1] label_indices, RefinementSearch refinement_search, bint uncovered,
                                  bint accumulated) nogil

    cdef Prediction* calculate_prediction(self, RefinementSearch refinement_search, bint uncovered,
                                          bint accumulated) nogil
