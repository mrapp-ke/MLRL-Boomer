from boomer.common._types cimport uint32
from boomer.common._indices cimport FullIndexVector, PartialIndexVector
from boomer.common.statistics cimport IStatistics, IStatisticsSubset

from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/head_refinement/prediction.h" nogil:

    cdef cppclass AbstractPrediction:

        # Functions:

        void apply(IStatistics& statistics, uint32 statisticIndex)


cdef extern from "cpp/head_refinement/prediction_evaluated.h" nogil:

    cdef cppclass AbstractEvaluatedPrediction(AbstractPrediction):
        pass


cdef extern from "cpp/head_refinement/head_refinement.h" nogil:

    cdef cppclass IHeadRefinement:

        const AbstractEvaluatedPrediction* findHead(AbstractEvaluatedPrediction* bestHead,
                                                    IStatisticsSubset& statisticsSubset, bool uncovered,
                                                    bool accumulated)

        unique_ptr[AbstractEvaluatedPrediction] pollHead()


cdef extern from "cpp/head_refinement/head_refinement_factory.h" nogil:

    cdef cppclass IHeadRefinementFactory:

        unique_ptr[IHeadRefinement] create(const FullIndexVector& labelIndices)

        unique_ptr[IHeadRefinement] create(const PartialIndexVector& labelIndices)


cdef extern from "cpp/head_refinement/head_refinement_single.h" nogil:

    cdef cppclass SingleLabelHeadRefinementFactoryImpl"SingleLabelHeadRefinementFactory"(IHeadRefinementFactory):
        pass


cdef extern from "cpp/head_refinement/head_refinement_full.h" nogil:

    cdef cppclass FullHeadRefinementFactoryImpl"FullHeadRefinementFactory"(IHeadRefinementFactory):
        pass


cdef class HeadRefinementFactory:

    # Attributes:

    cdef shared_ptr[IHeadRefinementFactory] head_refinement_factory_ptr


cdef class SingleLabelHeadRefinementFactory(HeadRefinementFactory):
    pass


cdef class FullHeadRefinementFactory(HeadRefinementFactory):
    pass
