from boomer.algorithm._model cimport uint8, intp, float32


cdef class InstanceSubSampling:

    # Functions:

    cdef uint8[::1] sub_sample(self, float32[::1, :] x)


cdef class FeatureSubSampling:

    # Functions:

    cdef intp[::1] sub_sample(self, float32[::1, :] x)
