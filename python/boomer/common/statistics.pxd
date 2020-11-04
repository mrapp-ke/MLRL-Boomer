from boomer.common._arrays cimport uint32
from boomer.common._indices cimport FullIndexVector, PartialIndexVector
from boomer.common._predictions cimport FullPrediction, PartialPrediction
from boomer.common.input_data cimport LabelMatrix

from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/statistics.h" nogil:

    cdef cppclass IStatisticsSubset:
        pass


    cdef cppclass AbstractDecomposableStatisticsSubset(IStatisticsSubset):
        pass


    cdef cppclass AbstractStatistics:

        # Functions:

        void resetSampledStatistics()

        void addSampledStatistic(uint32 statisticIndex, uint32 weight)

        uint32 getNumStatistics()

        uint32 getNumLabels()


cdef class StatisticsProvider:

    # Attributes:

    cdef shared_ptr[AbstractStatistics] statistics_ptr

    # Functions:

    cdef AbstractStatistics* get(self)

    cdef void switch_rule_evaluation(self)


cdef class StatisticsProviderFactory:

    # Functions:

    cdef StatisticsProvider create(self, LabelMatrix label_matrix)
