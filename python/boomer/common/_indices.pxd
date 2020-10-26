"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for classes that provide access to data that is stored in matrices or vectors.
"""
from boomer.common._arrays cimport uint32
from boomer.common._data cimport IRandomAccessVector
from boomer.common.sub_sampling cimport IWeightVector
from boomer.common.thresholds cimport AbstractThresholds, IThresholdsSubset

from libcpp.memory cimport unique_ptr


cdef extern from "cpp/indices.h" nogil:

    cdef cppclass IIndexVector(IRandomAccessVector[uint32]):

        # Functions:

        unique_ptr[IThresholdsSubset] createSubset(AbstractThresholds& thresholds, IWeightVector& weights)
