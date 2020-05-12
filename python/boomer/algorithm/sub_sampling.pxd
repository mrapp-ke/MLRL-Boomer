from boomer.algorithm._arrays cimport uint8, uint32, intp, float32
from boomer.algorithm._random cimport RNG


cdef class InstanceSubSampling:

    # Functions:

    cdef uint32[::1] sub_sample(self, intp num_examples, RNG rng)


cdef class Bagging(InstanceSubSampling):

    # Attributes:

    cdef readonly float32 sample_size

    # Functions:

    cdef uint32[::1] sub_sample(self, intp num_examples, RNG rng)


cdef class RandomInstanceSubsetSelection(InstanceSubSampling):

    # Attributes
    cdef readonly float32 sample_size

    # Functions:

    cdef uint32[::1] sub_sample(self, intp num_examples, RNG rng)


cdef class FeatureSubSampling:

    # Functions:

    cdef intp[::1] sub_sample(self, intp num_features, intp random_state)


cdef class RandomFeatureSubsetSelection(FeatureSubSampling):

    # Attributes:

    cdef readonly float32 sample_size

    # Functions:

    cdef intp[::1] sub_sample(self, intp num_features, intp random_state)


cdef class LabelSubSampling:

    # Functions:

    cdef intp[::1] sub_sample(self, intp num_labels, intp random_state)


cdef class RandomLabelSubsetSelection(LabelSubSampling):

    # Attributes:

    cdef readonly intp num_samples

    # Functions:

    cdef intp[::1] sub_sample(self, intp num_labels, intp random_state)
