from boomer.common._types cimport uint32
from boomer.common.input_data cimport LabelMatrix

from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/statistics/statistics_subset.h" nogil:

    cdef cppclass IStatisticsSubset:
        pass


cdef extern from "cpp/statistics/statistics.h" nogil:

    cdef cppclass IStatistics:

        # Functions:

        void resetSampledStatistics()

        void addSampledStatistic(uint32 statisticIndex, uint32 weight)

        uint32 getNumStatistics()

        uint32 getNumLabels()


cdef class StatisticsProvider:

    # Attributes:

    cdef shared_ptr[IStatistics] statistics_ptr

    # Functions:

    cdef IStatistics* get(self)

    cdef void switch_rule_evaluation(self)


cdef class StatisticsProviderFactory:

    # Functions:

    cdef StatisticsProvider create(self, LabelMatrix label_matrix)
