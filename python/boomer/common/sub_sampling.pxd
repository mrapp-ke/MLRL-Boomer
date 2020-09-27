from boomer.common._arrays cimport uint32, float32
from boomer.common._data cimport ISparseRandomAccessVector, IIndexVector
from boomer.common._random cimport RNG

from libcpp.pair cimport pair


cdef extern from "cpp/sub_sampling.h" nogil:

    cdef cppclass IWeightVector(ISparseRandomAccessVector[uint32]):

        # Functions:

        uint32 getSumOfWeights()


    cdef cppclass IInstanceSubSampling:

        # Functions:

        IWeightVector* subSample(uint32 numExamples, RNG* rng)


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

        IIndexVector* subSample(uint32 numFeatures, RNG* rng)


    cdef cppclass RandomFeatureSubsetSelectionImpl(IFeatureSubSampling):

        # Constructors:

        RandomFeatureSubsetSelectionImpl(float32 sampleSize) except +


    cdef cppclass NoFeatureSubSamplingImpl(IFeatureSubSampling):
        pass


    cdef cppclass ILabelSubSampling:

        # Functions:

        IIndexVector* subSample(uint32 numLabels, RNG* rng)


    cdef cppclass RandomLabelSubsetSelectionImpl(ILabelSubSampling):

        # Constructors:

        RandomLabelSubsetSelectionImpl(uint32 numSamples)


    cdef cppclass NoLabelSubSamplingImpl(ILabelSubSampling):
        pass


cdef class InstanceSubSampling:

    # Functions:

    cdef pair[uint32[::1], uint32] sub_sample(self, uint32 num_examples, RNG* rng)


cdef class Bagging(InstanceSubSampling):

    # Attributes:

    cdef readonly float32 sample_size

    # Functions:

    cdef pair[uint32[::1], uint32] sub_sample(self, uint32 num_examples, RNG* rng)


cdef class RandomInstanceSubsetSelection(InstanceSubSampling):

    # Attributes
    cdef readonly float32 sample_size

    # Functions:

    cdef pair[uint32[::1], uint32] sub_sample(self, uint32 num_examples, RNG* rng)


cdef class FeatureSubSampling:

    # Functions:

    cdef uint32[::1] sub_sample(self, uint32 num_features, RNG* rng)


cdef class RandomFeatureSubsetSelection(FeatureSubSampling):

    # Attributes:

    cdef readonly float32 sample_size

    # Functions:

    cdef uint32[::1] sub_sample(self, uint32 num_features, RNG* rng)


cdef class LabelSubSampling:

    # Functions:

    cdef uint32[::1] sub_sample(self, uint32 num_labels, RNG* rng)


cdef class RandomLabelSubsetSelection(LabelSubSampling):

    # Attributes:

    cdef readonly uint32 num_samples

    # Functions:

    cdef uint32[::1] sub_sample(self, uint32 num_labels, RNG* rng)
