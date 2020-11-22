from boomer.common._types cimport uint32, float32
from boomer.common._indices cimport IIndexVector
from boomer.common._random cimport RNG

from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/sampling/weight_vector.h" nogil:

    cdef cppclass IWeightVector:

        # Functions:

        bool hasZeroWeights()

        uint32 getSumOfWeights()

        uint32 getWeight(uint32 pos)


cdef extern from "cpp/sampling/instance_sampling.h" nogil:

    cdef cppclass IInstanceSubSampling:

        # Functions:

        unique_ptr[IWeightVector] subSample(uint32 numExamples, RNG& rng)


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

        # Functions:

        unique_ptr[IIndexVector] subSample(uint32 numFeatures, RNG& rng)


cdef extern from "cpp/sampling/feature_sampling_random.h" nogil:

    cdef cppclass RandomFeatureSubsetSelectionImpl"RandomFeatureSubsetSelection"(IFeatureSubSampling):

        # Constructors:

        RandomFeatureSubsetSelectionImpl(float32 sampleSize) except +


cdef extern from "cpp/sampling/feature_sampling_no.h" nogil:

    cdef cppclass NoFeatureSubSamplingImpl"NoFeatureSubSampling"(IFeatureSubSampling):
        pass


cdef extern from "cpp/sampling/label_sampling.h" nogil:

    cdef cppclass ILabelSubSampling:

        # Functions:

        unique_ptr[IIndexVector] subSample(uint32 numLabels, RNG& rng)


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
