from boomer.common._arrays cimport uint32
from boomer.common._predictions cimport PredictionCandidate
from boomer.common.statistics cimport AbstractStatisticsSubset

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/head_refinement.h" nogil:

    cdef cppclass IHeadRefinement:

        PredictionCandidate* findHead(PredictionCandidate* bestHead, PredictionCandidate* recyclableHead,
                                      const uint32* labelIndices, AbstractStatisticsSubset* statisticsSubset,
                                      bool uncovered, bool accumulated)

        PredictionCandidate* calculatePrediction(AbstractStatisticsSubset* statisticsSubset, bool uncovered,
                                                 bool accumulated)


    cdef cppclass SingleLabelHeadRefinementImpl(IHeadRefinement):
        pass


    cdef cppclass FullHeadRefinementImpl(IHeadRefinement):
        pass


cdef class HeadRefinement:

    # Attributes:

    cdef shared_ptr[IHeadRefinement] head_refinement_ptr


cdef class SingleLabelHeadRefinement(HeadRefinement):
    pass


cdef class FullHeadRefinement(HeadRefinement):
    pass
