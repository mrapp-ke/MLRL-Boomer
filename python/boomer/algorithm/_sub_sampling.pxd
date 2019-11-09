from boomer.algorithm._model cimport uint8, intp, float32


cdef class InstanceSubSampling:

    # Functions:

    cdef uint8[::1] sub_sample(self, float32[::1, :] x, int random_state)


cdef class Bagging(InstanceSubSampling):

    # Attributes:

    cdef readonly float sample_size

    cdef readonly bint with_replacement

    # Functions:

    cdef uint8[::1] sub_sample(self, float32[::1, :] x, int random_state)

    cdef __sub_sample_with_replacement(self, uint8[::1] weights, bint num_examples, bint num_samples, rng_randint)

    cdef __sub_sample_without_replacement(self, uint8[::1] weights, bint num_examples, bint num_samples, rng_randint)


cdef class FeatureSubSampling:

    # Functions:

    cdef intp[::1] sub_sample(self, float32[::1, :] x, int random_state)


cdef class RandomFeatureSubsetSelection(FeatureSubSampling):

    # Attributes:

    cdef readonly float sample_size

    # Functions:

    cdef intp[::1] sub_sample(self, float32[::1, :] x, int random_state)
