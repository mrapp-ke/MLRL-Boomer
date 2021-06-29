"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_shared


cdef class InstanceSamplingFactory:
    """
    A wrapper for the pure virtual C++ class `IInstanceSamplingFactory`.
    """
    pass


cdef class InstanceSamplingWithReplacementFactory(InstanceSamplingFactory):
    """
    A wrapper for the C++ class `InstanceSamplingWithReplacementFactory`.
    """

    def __cinit__(self, float32 sample_size = 1.0):
        """
        :param sample_size: The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
                            60 % of the available examples). Must be in (0, 1]
        """
        self.instance_sampling_factory_ptr = <shared_ptr[IInstanceSamplingFactory]>make_shared[InstanceSamplingWithReplacementFactoryImpl](
            sample_size)


cdef class InstanceSamplingWithoutReplacementFactory(InstanceSamplingFactory):
    """
    A wrapper for the C++ class `InstanceSamplingWithoutReplacementFactory`.
    """

    def __cinit__(self, float32 sample_size = 0.66):
        """
        param sample_size: The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
                           60 % of the available examples). Must be in (0, 1)
        """
        self.instance_sampling_factory_ptr = <shared_ptr[IInstanceSamplingFactory]>make_shared[InstanceSamplingWithoutReplacementFactoryImpl](
            sample_size)


cdef class LabelWiseStratifiedSamplingFactory(InstanceSamplingFactory):
    """
    A wrapper for the C++ class `LabelWiseStratifiedSamplingFactory`.
    """

    def __cinit__(self, float32 sample_size = 0.66):
        """
        param sample_size: The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
                           60 % of the available examples). Must be in (0, 1)
        """
        self.instance_sampling_factory_ptr = <shared_ptr[IInstanceSamplingFactory]>make_shared[LabelWiseStratifiedSamplingFactoryImpl](
            sample_size)


cdef class ExampleWiseStratifiedSamplingFactory(InstanceSamplingFactory):
    """
    A wrapper for the C++ class `ExampleWiseStratifiedSamplingFactory`.
    """

    def __cinit__(self, float32 sample_size = 0.66):
        """
        param sample_size: The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
                           60 % of the available examples). Must be in (0, 1)
        """
        self.instance_sampling_factory_ptr = <shared_ptr[IInstanceSamplingFactory]>make_shared[ExampleWiseStratifiedSamplingFactoryImpl](
            sample_size)


cdef class NoInstanceSamplingFactory(InstanceSamplingFactory):
    """
    A wrapper for the C++ class `NoInstanceSamplingFactory`.
    """

    def __cinit__(self):
        self.instance_sampling_factory_ptr = <shared_ptr[IInstanceSamplingFactory]>make_shared[NoInstanceSamplingFactoryImpl]()


cdef class FeatureSubSamplingFactory:
    """
    A wrapper for the pure virtual C++ class `IFeatureSubSamplingFactory`.
    """
    pass


cdef class RandomFeatureSubsetSelectionFactory(FeatureSubSamplingFactory):
    """
    A wrapper for the C++ class `RandomFeatureSubsetSelectionFactory`.
    """

    def __cinit__(self, float32 sample_size = 0.0):
        """
        :param sample_size: The fraction of features to be included in the sample (e.g. a value of 0.6 corresponds to
                            60 % of the available features). Must be in (0, 1) or 0, if the default sample size
                            `floor(log2(num_features - 1) + 1)` should be used
        """
        self.feature_sampling_factory_ptr = <shared_ptr[IFeatureSubSamplingFactory]>make_shared[RandomFeatureSubsetSelectionFactoryImpl](
            sample_size)


cdef class NoFeatureSubSamplingFactory(FeatureSubSamplingFactory):
    """
    A wrapper for the C++ class `NoFeatureSubSamplingFactory`.
    """

    def __cinit__(self):
        self.feature_sampling_factory_ptr = <shared_ptr[IFeatureSubSamplingFactory]>make_shared[NoFeatureSubSamplingFactoryImpl]()


cdef class LabelSamplingFactory:
    """
    A wrapper for the pure virtual C++ class `ILabelSamplingFactory`.
    """
    pass


cdef class RandomLabelSubsetSelectionFactory(LabelSamplingFactory):
    """
    A wrapper for the C++ class `RandomLabelSubsetSelectionFactory`.
    """

    def __cinit__(self, uint32 num_samples):
        """
        :param num_samples: The number of labels to be included in the sample
        """
        self.label_sampling_factory_ptr = <shared_ptr[ILabelSamplingFactory]>make_shared[RandomLabelSubsetSelectionFactoryImpl](
            num_samples)


cdef class NoLabelSamplingFactory(LabelSamplingFactory):
    """
    A wrapper for the C++ class `NoLabelSamplingFactory`.
    """

    def __cinit__(self):
        self.label_sampling_factory_ptr = <shared_ptr[ILabelSamplingFactory]>make_shared[NoLabelSamplingFactoryImpl]()


cdef class PartitionSamplingFactory:
    """
    A wrapper for the pure virtual C++ class `IPartitionSamplingFactory`.
    """
    pass


cdef class NoPartitionSamplingFactory(PartitionSamplingFactory):
    """
    A wrapper for hte C++ class `NoPartitionSamplingFactory`.
    """

    def __cinit__(self):
        self.partition_sampling_factory_ptr = <shared_ptr[IPartitionSamplingFactory]>make_shared[NoPartitionSamplingFactoryImpl]()


cdef class RandomBiPartitionSamplingFactory(PartitionSamplingFactory):
    """
    A wrapper for the C++ class `RandomBiPartitionSamplingFactory`.
    """

    def __cinit__(self, float32 holdout_set_size):
        """
        :param holdout_set_size: The fraction of examples to be included in the holdout set (e.g. a value of 0.6
                                 corresponds to 60 % of the available examples). Must be in (0, 1)
        """
        self.partition_sampling_factory_ptr = <shared_ptr[IPartitionSamplingFactory]>make_shared[RandomBiPartitionSamplingFactoryImpl](
            holdout_set_size)


cdef class ExampleWiseStratifiedBiPartitionSamplingFactory(PartitionSamplingFactory):
    """
    A wrapper for the C++ class `ExampleWiseStratifiedBiPartitionSamplingFactory`.
    """

    def __cinit__(self, float32 holdout_set_size):
        """
        :param holdout_set_size: The fraction of examples to be included in the holdout set (e.g. a value of 0.6
                                 corresponds to 60 % of the available examples). Must be in (0, 1)
        """
        self.partition_sampling_factory_ptr = <shared_ptr[IPartitionSamplingFactory]>make_shared[ExampleWiseStratifiedBiPartitionSamplingFactoryImpl](
            holdout_set_size)


cdef class LabelWiseStratifiedBiPartitionSamplingFactory(PartitionSamplingFactory):
    """
    A wrapper for the C++ class `LabelWiseStratifiedBiPartitionSamplingFactory`.
    """

    def __cinit__(self, float32 holdout_set_size):
        """
        :param holdout_set_size: The fraction of examples to be included in the holdout set (e.g. a value of 0.6
                                 corresponds to 60 % of the available examples). Must be in (0, 1)
        """
        self.partition_sampling_factory_ptr = <shared_ptr[IPartitionSamplingFactory]>make_shared[LabelWiseStratifiedBiPartitionSamplingFactoryImpl](
            holdout_set_size)
