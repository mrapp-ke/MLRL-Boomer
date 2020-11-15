from boomer.common._types cimport float64
from boomer.common.statistics cimport IStatistics


cdef extern from "cpp/statistics.h" namespace "seco" nogil:

    cdef cppclass ICoverageStatistics(IStatistics):

        # Functions:

        float64 getSumOfUncoveredLabels()
