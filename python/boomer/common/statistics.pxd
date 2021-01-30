from boomer.common._types cimport uint32
from boomer.common.input cimport IRandomAccessLabelMatrix

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


cdef extern from "cpp/statistics/statistics_provider.h" nogil:

    cdef cppclass IStatisticsProvider:

        # Functions:

        IStatistics& get()

        void switchRuleEvaluation()


    cdef cppclass IStatisticsProviderFactory:

        unique_ptr[IStatisticsProvider] create(shared_ptr[IRandomAccessLabelMatrix] labelMatrixPtr)


cdef class StatisticsProviderFactory:

    # Attributes:

    cdef shared_ptr[IStatisticsProviderFactory] statistics_provider_factory_ptr
