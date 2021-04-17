"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_shared


cdef class InstanceSubSamplingFactory:
    """
    A wrapper for the pure virtual C++ class `IInstanceSubSamplingFactory`.
    """
    pass


cdef class BaggingFactory(InstanceSubSamplingFactory):
    """
    A wrapper for the C++ class `BaggingFactory`.
    """

    def __cinit__(self, float32 sample_size = 1.0):
        """
        :param sample_size: The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
                            60 % of the available examples). Must be in (0, 1]
        """
        self.instance_sub_sampling_factory_ptr = <shared_ptr[IInstanceSubSamplingFactory]>make_shared[BaggingFactoryImpl](
            sample_size)


cdef class RandomInstanceSubsetSelectionFactory(InstanceSubSamplingFactory):
    """
    A wrapper for the C++ class `RandomInstanceSubsetSelectionFactory`.
    """

    def __cinit__(self, float32 sample_size = 0.66):
        """
        param sample_size: The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
                           60 % of the available examples). Must be in (0, 1)
        """
        self.instance_sub_sampling_factory_ptr = <shared_ptr[IInstanceSubSamplingFactory]>make_shared[RandomInstanceSubsetSelectionFactoryImpl](
            sample_size)


cdef class NoInstanceSubSamplingFactory(InstanceSubSamplingFactory):
    """
    A wrapper for the C++ class `NoInstanceSubSamplingFactory`.
    """

    def __cinit__(self):
        self.instance_sub_sampling_factory_ptr = <shared_ptr[IInstanceSubSamplingFactory]>make_shared[NoInstanceSubSamplingFactoryImpl]()


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
        self.feature_sub_sampling_factory_ptr = <shared_ptr[IFeatureSubSamplingFactory]>make_shared[RandomFeatureSubsetSelectionFactoryImpl](
            sample_size)


cdef class NoFeatureSubSamplingFactory(FeatureSubSamplingFactory):
    """
    A wrapper for the C++ class `NoFeatureSubSamplingFactory`.
    """

    def __cinit__(self):
        self.feature_sub_sampling_factory_ptr = <shared_ptr[IFeatureSubSamplingFactory]>make_shared[NoFeatureSubSamplingFactoryImpl]()


cdef class LabelSubSamplingFactory:
    """
    A wrapper for the pure virtual C++ class `ILabelSubSamplingFactory`.
    """
    pass


cdef class RandomLabelSubsetSelectionFactory(LabelSubSamplingFactory):
    """
    A wrapper for the C++ class `RandomLabelSubsetSelectionFactory`.
    """

    def __cinit__(self, uint32 num_samples):
        """
        :param num_samples: The number of labels to be included in the sample
        """
        self.label_sub_sampling_factory_ptr = <shared_ptr[ILabelSubSamplingFactory]>make_shared[RandomLabelSubsetSelectionFactoryImpl](
            num_samples)


cdef class NoLabelSubSamplingFactory(LabelSubSamplingFactory):
    """
    A wrapper for the C++ class `NoLabelSubSamplingFactory`.
    """

    def __cinit__(self):
        self.label_sub_sampling_factory_ptr = <shared_ptr[ILabelSubSamplingFactory]>make_shared[NoLabelSubSamplingFactoryImpl]()


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


cdef class BiPartitionSamplingFactory(PartitionSamplingFactory):
    """
    A wrapper for the C++ class `BiPartitionSamplingFactory`.
    """

    def __cinit__(self, float32 holdout_set_size):
        """
        :param holdout_set_size: The fraction of examples to be included in the holdout set (e.g. a value of 0.6
                                 corresponds to 60 % of the available examples). Must be in (0, 1)
        """
        self.partition_sampling_factory_ptr = <shared_ptr[IPartitionSamplingFactory]>make_shared[BiPartitionSamplingFactoryImpl](
            holdout_set_size)
