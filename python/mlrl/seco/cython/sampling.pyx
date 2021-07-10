"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport shared_ptr, make_shared


cdef class NoInstanceSamplingFactory(InstanceSamplingFactory):
    """
    A wrapper for the C++ class `NoInstanceSamplingFactory`.
    """

    def __cinit__(self):
        self.instance_sampling_factory_ptr = <shared_ptr[IInstanceSamplingFactory]>make_shared[NoInstanceSamplingFactoryImpl]()
