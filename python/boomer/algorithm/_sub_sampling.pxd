from boomer.algorithm._model cimport uint32, intp, float32
from boomer.algorithm._losses cimport Loss


cdef class InstanceSubSampling:

    # Functions:

    cdef uint32[::1] sub_sample(self, float32[::1, :] x, Loss loss, int random_state)


cdef class Bagging(InstanceSubSampling):

    # Attributes:

    cdef readonly float sample_size

    # Functions:

    cdef uint32[::1] sub_sample(self, float32[::1, :] x, Loss loss, int random_state)


cdef class RandomInstanceSubsetSelection(InstanceSubSampling):

    # Attributes
    cdef readonly float sample_size

    # Functions:

    cdef uint32[::1] sub_sample(self, float32[::1, :] x, Loss loss, int random_state)


cdef class FeatureSubSampling:

    # Functions:

    cdef intp[::1] sub_sample(self, float32[::1, :] x, int random_state)


cdef class RandomFeatureSubsetSelection(FeatureSubSampling):

    # Attributes:

    cdef readonly float sample_size

    # Functions:

    cdef intp[::1] sub_sample(self, float32[::1, :] x, int random_state)
