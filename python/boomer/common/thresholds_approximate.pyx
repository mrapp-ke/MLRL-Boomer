"""
@author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
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
