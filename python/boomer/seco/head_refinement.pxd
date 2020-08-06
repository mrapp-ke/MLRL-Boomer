from boomer.common._arrays cimport intp
from boomer.common.head_refinement cimport RefinementSearch
from boomer.common.rule_evaluation cimport Prediction
from boomer.common.head_refinement cimport HeadRefinement, HeadCandidate
from boomer.seco.lift_functions cimport AbstractLiftFunction

from libcpp.memory cimport shared_ptr


cdef class PartialHeadRefinement(HeadRefinement):

    # Attributes:

    cdef shared_ptr[AbstractLiftFunction] lift_function_ptr

    # Functions:

    cdef HeadCandidate* find_head(self, HeadCandidate* best_head, HeadCandidate* recyclable_head,
                                  intp[::1] label_indices, RefinementSearch refinement_search, bint uncovered,
                                  bint accumulated)

    cdef Prediction* calculate_prediction(self, RefinementSearch refinement_search, bint uncovered, bint accumulated)
