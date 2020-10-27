"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for classes that provide access to data that is stored in matrices or vectors.
"""
from boomer.common._arrays cimport uint32
from boomer.common.statistics cimport AbstractStatistics, IStatisticsSubset
from boomer.common.sub_sampling cimport IWeightVector
from boomer.common.thresholds cimport AbstractThresholds, IThresholdsSubset

from libcpp.memory cimport unique_ptr


cdef extern from "cpp/indices.h" nogil:

    cdef cppclass IIndexVector:

        # Functions:

        uint32 getNumElements()

        uint32 getIndex(uint32 pos)

        unique_ptr[IThresholdsSubset] createSubset(AbstractThresholds& thresholds, IWeightVector& weights)

        unique_ptr[IStatisticsSubset] createSubset(const AbstractStatistics& statistics)


    cdef cppclass RangeIndexVector(IIndexVector):
        pass


    cdef cppclass DenseIndexVector(IIndexVector):
        pass
