from libcpp.memory cimport make_shared, shared_ptr
from common.cython.sampling cimport InstanceSubSampling


cdef class GradientBasedLabelSet(InstanceSubSampling):
    """
    A wrapper for the C++ class 'GradientBasedLabelSet'.
    """

    def __cinit__(self, float32 sample_size = 0.6):
        self.instance_sub_sampling_ptr = <shared_ptr[IInstanceSubSampling]>make_shared[GradientBasedLabelSetImpl](sample_size)


cdef class GradientBasedLabelWise(InstanceSubSampling):
    """
    A wrapper for the C++ class 'GradientBasedLabelWise'.
    """

    def __cinit__(self, float32 sample_size = 0.6):
        self.instance_sub_sampling_ptr = <shared_ptr[IInstanceSubSampling]>make_shared[GradientBasedLabelWiseImpl](sample_size)


cdef class IterativeStratificationLabelWise(InstanceSubSampling):
    """
    A wrapper for the C++ class 'IterativeStratificationLabelWise'.
    """

    def __cinit__(self, float32 sample_size = 0.6):
        self.instance_sub_sampling_ptr = <shared_ptr[IInstanceSubSampling]>make_shared[IterativeStratificationLabelWiseImpl](sample_size)


cdef class IterativeStratificationLabelSet(InstanceSubSampling):
    """
    A wrapper for the C++ class 'IterativeStratificationLabelSet'.
    """

    def __cinit__(self, float32 sample_size = 0.6):
        self.instance_sub_sampling_ptr = <shared_ptr[IInstanceSubSampling]>make_shared[IterativeStratificationLabelSetImpl](sample_size)
