from boomer.common._types cimport float64
from boomer.common.statistics cimport IStatistics


cdef extern from "cpp/statistics.h" namespace "seco" nogil:

    cdef cppclass AbstractCoverageStatistics(IStatistics):

        # Attributes:

        float64 sumUncoveredLabels_;
