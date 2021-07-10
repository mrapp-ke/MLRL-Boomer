from mlrl.common.cython.sampling cimport IInstanceSamplingFactory, InstanceSamplingFactory


cdef extern from "seco/sampling/instance_sampling_no.hpp" namespace "seco" nogil:

    cdef cppclass NoInstanceSamplingFactoryImpl"seco::NoInstanceSamplingFactory"(IInstanceSamplingFactory):
        pass


cdef class NoInstanceSamplingFactory(InstanceSamplingFactory):
    pass
