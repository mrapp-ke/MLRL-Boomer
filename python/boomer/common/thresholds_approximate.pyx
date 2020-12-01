"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for classes that provide access to approximate thresholds that may be used by the conditions of
rules.
"""
from boomer.common.binning cimport FeatureBinning

from libcpp.memory cimport make_shared


cdef class ApproximateThresholdsFactory(ThresholdsFactory):
    """
    A wrapper for the C++ class `ApproximateThresholdsFactory`.
    """

    def __cinit__(self, FeatureBinning binning):
        self.thresholds_factory_ptr = <shared_ptr[IThresholdsFactory]>make_shared[ApproximateThresholdsFactoryImpl](
            binning.binning_ptr)
