from mlrl.common.cython._types cimport uint32, float32

from libcpp.memory cimport shared_ptr


cdef extern from "common/sampling/random.hpp" nogil:

    cdef cppclass RNG:

        # Constructors:

        RNG(uint32 randomState) except +


cdef extern from "common/sampling/weight_vector.hpp" nogil:

    cdef cppclass IWeightVector:
        pass


cdef extern from "common/sampling/instance_sampling.hpp" nogil:

    cdef cppclass IInstanceSamplingFactory:
        pass


cdef extern from "common/sampling/feature_sampling.hpp" nogil:

    cdef cppclass IFeatureSamplingFactory:
        pass


cdef extern from "common/sampling/feature_sampling_without_replacement.hpp" nogil:

    cdef cppclass FeatureSamplingWithoutReplacementFactoryImpl"FeatureSamplingWithoutReplacementFactory"(
            IFeatureSamplingFactory):

        # Constructors

        FeatureSamplingWithoutReplacementFactoryImpl(float32 sampleSize) except +


cdef extern from "common/sampling/feature_sampling_no.hpp" nogil:

    cdef cppclass NoFeatureSamplingFactoryImpl"NoFeatureSamplingFactory"(IFeatureSamplingFactory):
        pass


cdef extern from "common/sampling/label_sampling.hpp" nogil:

    cdef cppclass ILabelSamplingFactory:
        pass


cdef extern from "common/sampling/label_sampling_without_replacement.hpp" nogil:

    cdef cppclass LabelSamplingWithoutReplacementFactoryImpl"LabelSamplingWithoutReplacementFactory"(
            ILabelSamplingFactory):

        # Constructors:

        LabelSamplingWithoutReplacementFactoryImpl(uint32 numSamples) except +


cdef extern from "common/sampling/label_sampling_no.hpp" nogil:

    cdef cppclass NoLabelSamplingFactoryImpl"NoLabelSamplingFactory"(ILabelSamplingFactory):
        pass


cdef extern from "common/sampling/partition_sampling.hpp" nogil:

    cdef cppclass IPartitionSamplingFactory:
        pass


cdef extern from "common/sampling/partition_sampling_no.hpp" nogil:

    cdef cppclass NoPartitionSamplingFactoryImpl"NoPartitionSamplingFactory"(IPartitionSamplingFactory):
        pass


cdef extern from "common/sampling/partition_sampling_bi_random.hpp" nogil:

    cdef cppclass RandomBiPartitionSamplingFactoryImpl"RandomBiPartitionSamplingFactory"(IPartitionSamplingFactory):

        # Constructors:

        RandomBiPartitionSamplingFactoryImpl(float32 holdout_set_size) except +


cdef extern from "common/sampling/partition_sampling_bi_stratified_example_wise.hpp" nogil:

    cdef cppclass ExampleWiseStratifiedBiPartitionSamplingFactoryImpl"ExampleWiseStratifiedBiPartitionSamplingFactory"(
            IPartitionSamplingFactory):

        # Constructors:

        ExampleWiseStratifiedBiPartitionSamplingFactoryImpl(float32 holdout_set_size) except +


cdef extern from "common/sampling/partition_sampling_bi_stratified_label_wise.hpp" nogil:

    cdef cppclass LabelWiseStratifiedBiPartitionSamplingFactoryImpl"LabelWiseStratifiedBiPartitionSamplingFactory"(
            IPartitionSamplingFactory):

        # Constructors:

        LabelWiseStratifiedBiPartitionSamplingFactoryImpl(float32 holdout_set_size) except +


cdef class InstanceSamplingFactory:

    # Attributes:

    cdef shared_ptr[IInstanceSamplingFactory] instance_sampling_factory_ptr


cdef class FeatureSamplingFactory:

    # Attributes:

    cdef shared_ptr[IFeatureSamplingFactory] feature_sampling_factory_ptr


cdef class FeatureSamplingWithoutReplacementFactory(FeatureSamplingFactory):
    pass


cdef class NoFeatureSamplingFactory(FeatureSamplingFactory):
    pass


cdef class LabelSamplingFactory:

    # Attributes:

    cdef shared_ptr[ILabelSamplingFactory] label_sampling_factory_ptr


cdef class LabelSamplingWithoutReplacementFactory(LabelSamplingFactory):
    pass


cdef class NoLabelSamplingFactory(LabelSamplingFactory):
    pass


cdef class PartitionSamplingFactory:

    # Attributes:

    cdef shared_ptr[IPartitionSamplingFactory] partition_sampling_factory_ptr


cdef class NoPartitionSamplingFactory(PartitionSamplingFactory):
    pass


cdef class RandomBiPartitionSamplingFactory(PartitionSamplingFactory):
    pass


cdef class ExampleWiseStratifiedBiPartitionSamplingFactory(PartitionSamplingFactory):
    pass


cdef class LabelWiseStratifiedBiPartitionSamplingFactory(PartitionSamplingFactory):
    pass
