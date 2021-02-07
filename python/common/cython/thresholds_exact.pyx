"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport shared_ptr, make_shared


cdef class ExactThresholdsFactory(ThresholdsFactory):
    """
    A wrapper for the C++ class `ExactThresholdsFactory`.
    """

    def __cinit__(self):
        self.thresholds_factory_ptr = <shared_ptr[IThresholdsFactory]>make_shared[ExactThresholdsFactoryImpl]()
