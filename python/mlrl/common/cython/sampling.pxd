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


cdef extern from "common/sampling/instance_sampling_with_replacement.hpp" nogil:

    cdef cppclass InstanceSamplingWithReplacementFactoryImpl"InstanceSamplingWithReplacementFactory"(
            IInstanceSamplingFactory):

        # Constructors:

        InstanceSamplingWithReplacementFactoryImpl(float32 sampleSize) except +


cdef extern from "common/sampling/instance_sampling_without_replacement.hpp" nogil:

    cdef cppclass InstanceSamplingWithoutReplacementFactoryImpl"InstanceSamplingWithoutReplacementFactory"(
            IInstanceSamplingFactory):

        # Constructors:

        InstanceSamplingWithoutReplacementFactoryImpl(float32 sampleSize) except +


cdef extern from "common/sampling/instance_sampling_stratified_label_wise.hpp" nogil:

    cdef cppclass LabelWiseStratifiedSamplingFactoryImpl"LabelWiseStratifiedSamplingFactory"(IInstanceSamplingFactory):

        # Constructors:

        LabelWiseStratifiedSamplingFactoryImpl(float32 sampleSize) except +


cdef extern from "common/sampling/instance_sampling_stratified_example_wise.hpp" nogil:

    cdef cppclass ExampleWiseStratifiedSamplingFactoryImpl"ExampleWiseStratifiedSamplingFactory"(
            IInstanceSamplingFactory):

        # Constructors:

        ExampleWiseStratifiedSamplingFactoryImpl(float32 sampleSize) except +


cdef extern from "common/sampling/instance_sampling_no.hpp" nogil:

    cdef cppclass NoInstanceSamplingFactoryImpl"NoInstanceSamplingFactory"(IInstanceSamplingFactory):
        pass


cdef extern from "common/sampling/feature_sampling.hpp" nogil:

    cdef cppclass IFeatureSubSamplingFactory:
        pass


cdef extern from "common/sampling/feature_sampling_random.hpp" nogil:

    cdef cppclass RandomFeatureSubsetSelectionFactoryImpl"RandomFeatureSubsetSelectionFactory"(
            IFeatureSubSamplingFactory):

        # Constructors

        RandomFeatureSubsetSelectionFactoryImpl(float32 sampleSize) except +


cdef extern from "common/sampling/feature_sampling_no.hpp" nogil:

    cdef cppclass NoFeatureSubSamplingFactoryImpl"NoFeatureSubSamplingFactory"(IFeatureSubSamplingFactory):
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


cdef class InstanceSamplingWithReplacementFactory(InstanceSamplingFactory):
    pass


cdef class InstanceSamplingWithoutReplacementFactory(InstanceSamplingFactory):
    pass


cdef class LabelWiseStratifiedSamplingFactory(InstanceSamplingFactory):
    pass


cdef class ExampleWiseStratifiedSamplingFactory(InstanceSamplingFactory):
    pass


cdef class NoInstanceSamplingFactory(InstanceSamplingFactory):
    pass


cdef class FeatureSubSamplingFactory:

    # Attributes:

    cdef shared_ptr[IFeatureSubSamplingFactory] feature_sampling_factory_ptr


cdef class RandomFeatureSubsetSelectionFactory(FeatureSubSamplingFactory):
    pass


cdef class NoFeatureSubSamplingFactory(FeatureSubSamplingFactory):
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
