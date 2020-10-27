from boomer.common._indices cimport RangeIndexVector, DenseIndexVector
from boomer.common._rule_evaluation cimport EvaluatedPrediction
from boomer.common._predictions cimport PredictionCandidate
from boomer.common.statistics cimport IStatisticsSubset

from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/head_refinement.h" nogil:

    cdef cppclass IHeadRefinement:

        const PredictionCandidate* findHead(PredictionCandidate* bestHead, IStatisticsSubset& statisticsSubset,
                                            bool uncovered, bool accumulated)

        unique_ptr[PredictionCandidate] pollHead()

        EvaluatedPrediction& calculatePrediction(IStatisticsSubset& statisticsSubset, bool uncovered, bool accumulated)


    cdef cppclass IHeadRefinementFactory:

        unique_ptr[IHeadRefinement] create(const RangeIndexVector& labelIndices)

        unique_ptr[IHeadRefinement] create(const DenseIndexVector& labelIndices)


    cdef cppclass SingleLabelHeadRefinementFactoryImpl(IHeadRefinementFactory):
        pass


    cdef cppclass FullHeadRefinementFactoryImpl(IHeadRefinementFactory):
        pass


cdef class HeadRefinementFactory:

    # Attributes:

    cdef shared_ptr[IHeadRefinementFactory] head_refinement_factory_ptr


cdef class SingleLabelHeadRefinementFactory(HeadRefinementFactory):
    pass


cdef class FullHeadRefinementFactory(HeadRefinementFactory):
    pass
