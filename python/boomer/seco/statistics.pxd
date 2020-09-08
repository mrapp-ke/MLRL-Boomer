from boomer.common._arrays cimport float64
from boomer.common.statistics cimport AbstractStatistics


cdef extern from "cpp/statistics.h" namespace "seco" nogil:

    cdef cppclass AbstractCoverageStatistics(AbstractStatistics):

        # Attributes:

        float64 sumUncoveredLabels_;
