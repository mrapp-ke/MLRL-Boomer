from boomer.common._arrays cimport intp
from boomer.common._predictions cimport PredictionCandidate
from boomer.common.head_refinement cimport AbstractRefinementSearch, HeadRefinement, HeadCandidate
from boomer.seco.lift_functions cimport AbstractLiftFunction

from libcpp.memory cimport shared_ptr


cdef class PartialHeadRefinement(HeadRefinement):

    # Attributes:

    cdef shared_ptr[AbstractLiftFunction] lift_function_ptr

    # Functions:

    cdef HeadCandidate* find_head(self, HeadCandidate* best_head, HeadCandidate* recyclable_head,
                                  const intp* label_indices, AbstractRefinementSearch* refinement_search,
                                  bint uncovered, bint accumulated) nogil

    cdef PredictionCandidate* calculate_prediction(self, AbstractRefinementSearch* refinement_search, bint uncovered,
                                                   bint accumulated) nogil
