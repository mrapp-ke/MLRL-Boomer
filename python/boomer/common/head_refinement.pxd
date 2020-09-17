from boomer.common._arrays cimport uint32
from boomer.common._predictions cimport PredictionCandidate
from boomer.common.statistics cimport AbstractRefinementSearch

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


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

    # Attributes:

    cdef shared_ptr[AbstractHeadRefinement] head_refinement_ptr


cdef class SingleLabelHeadRefinement(HeadRefinement):
    pass


cdef class FullHeadRefinement(HeadRefinement):
    pass
