from boomer.common._types cimport uint32, float32
from boomer.common._indices cimport IIndexVector
from boomer.common._random cimport RNG

from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/sub_sampling.h" nogil:

    cdef cppclass IWeightVector:

        # Functions:

        bool hasZeroWeights()

        uint32 getSumOfWeights()

        uint32 getWeight(uint32 pos)


    cdef cppclass IInstanceSubSampling:

        # Functions:

        unique_ptr[IWeightVector] subSample(uint32 numExamples, RNG& rng)


    cdef cppclass BaggingImpl(IInstanceSubSampling):

        # Constructors:

        BaggingImpl(float32 sampleSize) except +


    cdef cppclass RandomInstanceSubsetSelectionImpl(IInstanceSubSampling):

        # Constructors:

        RandomInstanceSubsetSelectionImpl(float32 sampleSize)


    cdef cppclass NoInstanceSubSamplingImpl(IInstanceSubSampling):
        pass


    cdef cppclass IFeatureSubSampling:

        # Functions:

        unique_ptr[IIndexVector] subSample(uint32 numFeatures, RNG& rng)


    cdef cppclass RandomFeatureSubsetSelectionImpl(IFeatureSubSampling):

        # Constructors:

        RandomFeatureSubsetSelectionImpl(float32 sampleSize) except +


    cdef cppclass NoFeatureSubSamplingImpl(IFeatureSubSampling):
        pass


    cdef cppclass ILabelSubSampling:

        # Functions:

        unique_ptr[IIndexVector] subSample(uint32 numLabels, RNG& rng)


    cdef cppclass RandomLabelSubsetSelectionImpl(ILabelSubSampling):

        # Constructors:

        RandomLabelSubsetSelectionImpl(uint32 numSamples)


    cdef cppclass NoLabelSubSamplingImpl(ILabelSubSampling):
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
