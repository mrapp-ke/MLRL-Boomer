from boomer.common._indices cimport FullIndexVector, PartialIndexVector
from boomer.common._rule_evaluation cimport EvaluatedPrediction
from boomer.common._predictions cimport AbstractEvaluatedPrediction
from boomer.common.statistics cimport IStatisticsSubset

from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/head_refinement.h" nogil:

    cdef cppclass IHeadRefinement:

        const AbstractEvaluatedPrediction* findHead(AbstractEvaluatedPrediction* bestHead,
                                                    IStatisticsSubset& statisticsSubset, bool uncovered,
                                                    bool accumulated)

        unique_ptr[AbstractEvaluatedPrediction] pollHead()


    cdef cppclass IHeadRefinementFactory:

        unique_ptr[IHeadRefinement] create(const FullIndexVector& labelIndices)

        unique_ptr[IHeadRefinement] create(const PartialIndexVector& labelIndices)


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
