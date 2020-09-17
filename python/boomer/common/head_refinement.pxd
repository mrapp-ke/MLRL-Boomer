from boomer.common._arrays cimport uint32
from boomer.common._predictions cimport PredictionCandidate
from boomer.common.statistics cimport AbstractRefinementSearch

from libcpp cimport bool


cdef extern from "cpp/head_refinement.h" nogil:

    cdef cppclass AbstractHeadRefinement:

        PredictionCandidate* findHead(PredictionCandidate* bestHead, PredictionCandidate* recyclableHead,
                                      const uint32* labelIndices, AbstractRefinementSearch* refinementSearch,
                                      bool uncovered, bool accumulated)

        PredictionCandidate* calculatePrediction(AbstractRefinementSearch* refinementSearch, bool uncovered,
                                                 bool accumulated)


    cdef cppclass SingleLabelHeadRefinementImpl(AbstractHeadRefinement):
        pass


    cdef cppclass FullHeadRefinementImpl(AbstractHeadRefinement):
        pass


cdef class HeadRefinement:

    # Functions:

    cdef PredictionCandidate* find_head(self, PredictionCandidate* best_head, PredictionCandidate* recyclable_head,
                                        const uint32* label_indices, AbstractRefinementSearch* refinement_search,
                                        bint uncovered, bint accumulated) nogil

    cdef PredictionCandidate* calculate_prediction(self, AbstractRefinementSearch* refinement_search, bint uncovered,
                                                   bint accumulated) nogil


cdef class SingleLabelHeadRefinement(HeadRefinement):

    # Functions:

    cdef PredictionCandidate* find_head(self, PredictionCandidate* best_head, PredictionCandidate* recyclable_head,
                                        const uint32* label_indices, AbstractRefinementSearch* refinement_search,
                                        bint uncovered, bint accumulated) nogil

    cdef PredictionCandidate* calculate_prediction(self, AbstractRefinementSearch* refinement_search, bint uncovered,
                                                   bint accumulated) nogil


cdef class FullHeadRefinement(HeadRefinement):

    # Functions:

    cdef PredictionCandidate* find_head(self, PredictionCandidate* best_head, PredictionCandidate* recyclable_head,
                                        const uint32* label_indices, AbstractRefinementSearch* refinement_search,
                                        bint uncovered, bint accumulated) nogil

    cdef PredictionCandidate* calculate_prediction(self, AbstractRefinementSearch* refinement_search, bint uncovered,
                                                   bint accumulated) nogil
