from boomer.common.input cimport IRandomAccessLabelMatrix

from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/statistics/statistics.h" nogil:

    cdef cppclass IStatistics:
        pass


cdef extern from "cpp/statistics/statistics_provider.h" nogil:

    cdef cppclass IStatisticsProvider:

        # Functions:

        IStatistics& get()


cdef extern from "cpp/statistics/statistics_provider_factory.h" nogil:

    cdef cppclass IStatisticsProviderFactory:

        unique_ptr[IStatisticsProvider] create(shared_ptr[IRandomAccessLabelMatrix] labelMatrixPtr)


cdef class StatisticsProviderFactory:

    # Attributes:

    cdef shared_ptr[IStatisticsProviderFactory] statistics_provider_factory_ptr
