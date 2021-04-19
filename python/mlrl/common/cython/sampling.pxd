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

    cdef cppclass IInstanceSubSamplingFactory:
        pass


cdef extern from "common/sampling/instance_sampling_bagging.hpp" nogil:

    cdef cppclass BaggingFactoryImpl"BaggingFactory"(IInstanceSubSamplingFactory):

        # Constructors:

        BaggingFactoryImpl(float32 sampleSize) except +


cdef extern from "common/sampling/instance_sampling_random.hpp" nogil:

    cdef cppclass RandomInstanceSubsetSelectionFactoryImpl"RandomInstanceSubsetSelectionFactory"(
            IInstanceSubSamplingFactory):

        # Constructors:

        RandomInstanceSubsetSelectionFactoryImpl(float32 sampleSize) except +


cdef extern from "common/sampling/instance_sampling_stratified_label_wise.hpp" nogil:

    cdef cppclass LabelWiseStratifiedSamplingFactoryImpl"LabelWiseStratifiedSamplingFactory"(
            IInstanceSubSamplingFactory):

        # Constructors:

        LabelWiseStratifiedSamplingFactoryImpl(float32 sampleSize) except +


cdef extern from "common/sampling/instance_sampling_stratified_example_wise.hpp" nogil:

    cdef cppclass ExampleWiseStratifiedSamplingFactoryImpl"ExampleWiseStratifiedSamplingFactory"(
            IInstanceSubSamplingFactory):

        # Constructors:

        ExampleWiseStratifiedSamplingFactoryImpl(float32 sampleSize) except +


cdef extern from "common/sampling/instance_sampling_no.hpp" nogil:

    cdef cppclass NoInstanceSubSamplingFactoryImpl"NoInstanceSubSamplingFactory"(IInstanceSubSamplingFactory):
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

    cdef cppclass ILabelSubSamplingFactory:
        pass


cdef extern from "common/sampling/label_sampling_random.hpp" nogil:

    cdef cppclass RandomLabelSubsetSelectionFactoryImpl"RandomLabelSubsetSelectionFactory"(ILabelSubSamplingFactory):

        # Constructors:

        RandomLabelSubsetSelectionFactoryImpl(uint32 numSamples) except +


cdef extern from "common/sampling/label_sampling_no.hpp" nogil:

    cdef cppclass NoLabelSubSamplingFactoryImpl"NoLabelSubSamplingFactory"(ILabelSubSamplingFactory):
        pass


cdef extern from "common/sampling/partition_sampling.hpp" nogil:

    cdef cppclass IPartitionSamplingFactory:
        pass


cdef extern from "common/sampling/partition_sampling_no.hpp" nogil:

    cdef cppclass NoPartitionSamplingFactoryImpl"NoPartitionSamplingFactory"(IPartitionSamplingFactory):
        pass


cdef extern from "common/sampling/partition_sampling_bi.hpp" nogil:

    cdef cppclass BiPartitionSamplingFactoryImpl"BiPartitionSamplingFactory"(IPartitionSamplingFactory):

        # Constructors:

        BiPartitionSamplingFactoryImpl(float32 holdout_set_size) except +


cdef class InstanceSubSamplingFactory:

    # Attributes:

    cdef shared_ptr[IInstanceSubSamplingFactory] instance_sub_sampling_factory_ptr


cdef class BaggingFactory(InstanceSubSamplingFactory):
    pass


cdef class RandomInstanceSubsetSelectionFactory(InstanceSubSamplingFactory):
    pass


cdef class LabelWiseStratifiedSamplingFactory(InstanceSubSamplingFactory):
    pass


cdef class ExampleWiseStratifiedSamplingFactory(InstanceSubSamplingFactory):
    pass


cdef class NoInstanceSubSamplingFactory(InstanceSubSamplingFactory):
    pass


cdef class FeatureSubSamplingFactory:

    # Attributes:

    cdef shared_ptr[IFeatureSubSamplingFactory] feature_sub_sampling_factory_ptr


cdef class RandomFeatureSubsetSelectionFactory(FeatureSubSamplingFactory):
    pass


cdef class NoFeatureSubSamplingFactory(FeatureSubSamplingFactory):
    pass


cdef class LabelSubSamplingFactory:

    # Attributes:

    cdef shared_ptr[ILabelSubSamplingFactory] label_sub_sampling_factory_ptr


cdef class RandomLabelSubsetSelectionFactory(LabelSubSamplingFactory):
    pass


cdef class NoLabelSubSamplingFactory(LabelSubSamplingFactory):
    pass


cdef class PartitionSamplingFactory:

    # Attributes:

    cdef shared_ptr[IPartitionSamplingFactory] partition_sampling_factory_ptr


cdef class NoPartitionSamplingFactory(PartitionSamplingFactory):
    pass


cdef class BiPartitionSamplingFactory(PartitionSamplingFactory):
    pass
