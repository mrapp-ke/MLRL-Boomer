from boomer.common._types cimport uint32, float32

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/sampling/random.h" nogil:

    cdef cppclass RNG:

        # Constructors:

        RNG(uint32 randomState) except +


cdef extern from "cpp/sampling/weight_vector.h" nogil:

    cdef cppclass IWeightVector:
        pass


cdef extern from "cpp/sampling/instance_sampling.h" nogil:

    cdef cppclass IInstanceSubSampling:
        pass


cdef extern from "cpp/sampling/instance_sampling_bagging.h" nogil:

    cdef cppclass BaggingImpl"Bagging"(IInstanceSubSampling):

        # Constructors:

        BaggingImpl(float32 sampleSize) except +


cdef extern from "cpp/sampling/instance_sampling_random.h" nogil:

    cdef cppclass RandomInstanceSubsetSelectionImpl"RandomInstanceSubsetSelection"(IInstanceSubSampling):

        # Constructors:

        RandomInstanceSubsetSelectionImpl(float32 sampleSize)


cdef extern from "cpp/sampling/instance_sampling_no.h" nogil:

    cdef cppclass NoInstanceSubSamplingImpl"NoInstanceSubSampling"(IInstanceSubSampling):
        pass


cdef extern from "cpp/sampling/feature_sampling.h" nogil:

    cdef cppclass IFeatureSubSampling:
        pass


cdef extern from "cpp/sampling/feature_sampling_random.h" nogil:

    cdef cppclass RandomFeatureSubsetSelectionImpl"RandomFeatureSubsetSelection"(IFeatureSubSampling):

        # Constructors:

        RandomFeatureSubsetSelectionImpl(float32 sampleSize) except +


cdef extern from "cpp/sampling/feature_sampling_no.h" nogil:

    cdef cppclass NoFeatureSubSamplingImpl"NoFeatureSubSampling"(IFeatureSubSampling):
        pass


cdef extern from "cpp/sampling/label_sampling.h" nogil:

    cdef cppclass ILabelSubSampling:
        pass


cdef extern from "cpp/sampling/label_sampling_random.h" nogil:

    cdef cppclass RandomLabelSubsetSelectionImpl"RandomLabelSubsetSelection"(ILabelSubSampling):

        # Constructors:

        RandomLabelSubsetSelectionImpl(uint32 numSamples)


cdef extern from "cpp/sampling/label_sampling_no.h" nogil:

    cdef cppclass NoLabelSubSamplingImpl"NoLabelSubSampling"(ILabelSubSampling):
        pass


cdef class InstanceSubSampling:

    # Attributes:

    cdef shared_ptr[IInstanceSubSampling] instance_sub_sampling_ptr


cdef class Bagging(InstanceSubSampling):
    pass


cdef class RandomInstanceSubsetSelection(InstanceSubSampling):
    pass


cdef class NoInstanceSubSampling(InstanceSubSampling):
    pass


cdef class FeatureSubSampling:

    # Attributes:

    cdef shared_ptr[IFeatureSubSampling] feature_sub_sampling_ptr


cdef class RandomFeatureSubsetSelection(FeatureSubSampling):
    pass


cdef class NoFeatureSubSampling(FeatureSubSampling):
    pass


cdef class LabelSubSampling:

    # Attributes:

    cdef shared_ptr[ILabelSubSampling] label_sub_sampling_ptr


cdef class RandomLabelSubsetSelection(LabelSubSampling):
    pass


cdef class NoLabelSubSampling(LabelSubSampling):
    pass
