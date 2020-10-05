from boomer.common._arrays cimport uint32
from boomer.common._predictions cimport PredictionCandidate
from boomer.common.statistics cimport IStatisticsSubset

from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/head_refinement.h" nogil:

    cdef cppclass IHeadRefinement:

        bool findHead(PredictionCandidate* bestHead, unique_ptr[PredictionCandidate]& headPtr,
                      const uint32* labelIndices, IStatisticsSubset& statisticsSubset, bool uncovered, bool accumulated)

        PredictionCandidate& calculatePrediction(IStatisticsSubset& statisticsSubset, bool uncovered, bool accumulated)


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
