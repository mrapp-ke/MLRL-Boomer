from boomer.common._arrays cimport uint32
from boomer.common._predictions cimport PredictionCandidate
from boomer.common.head_refinement cimport AbstractRefinementSearch, HeadRefinement, AbstractHeadRefinement
from boomer.seco.lift_functions cimport AbstractLiftFunction

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/head_refinement.h" nogil:

    cdef cppclass PartialHeadRefinementImpl(AbstractHeadRefinement):
        pass


cdef class PartialHeadRefinement(HeadRefinement):

    # Attributes:

    cdef shared_ptr[AbstractLiftFunction] lift_function_ptr

    # Functions:

    cdef PredictionCandidate* find_head(self, PredictionCandidate* best_head, PredictionCandidate* recyclable_head,
                                        const uint32* label_indices, AbstractRefinementSearch* refinement_search,
                                        bint uncovered, bint accumulated) nogil

    cdef PredictionCandidate* calculate_prediction(self, AbstractRefinementSearch* refinement_search, bint uncovered,
                                                   bint accumulated) nogil
