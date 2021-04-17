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


cdef class FeatureSubSampling:
    """
    A wrapper for the pure virtual C++ class `IFeatureSubSampling`.
    """
    pass


cdef class RandomFeatureSubsetSelection(FeatureSubSampling):
    """
    A wrapper for the C++ class `RandomFeatureSubsetSelection`.
    """

    def __cinit__(self, float32 sample_size = 0.0):
        """
        :param sample_size: The fraction of features to be included in the sample (e.g. a value of 0.6 corresponds to
                            60 % of the available features). Must be in (0, 1) or 0, if the default sample size
                            `floor(log2(num_features - 1) + 1)` should be used
        """
        self.feature_sub_sampling_ptr = <shared_ptr[IFeatureSubSampling]>make_shared[RandomFeatureSubsetSelectionImpl](
            sample_size)


cdef class NoFeatureSubSampling(FeatureSubSampling):
    """
    A wrapper for the C++ class `NoFeatureSubSampling`.
    """

    def __cinit__(self):
        self.feature_sub_sampling_ptr = <shared_ptr[IFeatureSubSampling]>make_shared[NoFeatureSubSamplingImpl]()


cdef class LabelSubSampling:
    """
    A wrapper for the pure virtual C++ class `ILabelSubSampling`.
    """
    pass


cdef class RandomLabelSubsetSelection(LabelSubSampling):
    """
    A wrapper for the C++ class `RandomLabelSubsetSelection`.
    """

    def __cinit__(self, uint32 num_samples):
        """
        :param num_samples: The number of labels to be included in the sample
        """
        self.label_sub_sampling_ptr = <shared_ptr[ILabelSubSampling]>make_shared[RandomLabelSubsetSelectionImpl](
            num_samples)


cdef class NoLabelSubSampling(LabelSubSampling):
    """
    A wrapper for the C++ class `NoLabelSubSampling`.
    """

    def __cinit__(self):
        self.label_sub_sampling_ptr = <shared_ptr[ILabelSubSampling]>make_shared[NoLabelSubSamplingImpl]()


cdef class PartitionSampling:
    """
    A wrapper for the pure virtual C++ class `IPartitionSampling`.
    """
    pass


cdef class NoPartitionSampling(PartitionSampling):
    """
    A wrapper for the C++ class `NoPartitionSampling`.
    """

    def __cinit__(self):
        self.partition_sampling_ptr = <shared_ptr[IPartitionSampling]>make_shared[NoPartitionSamplingImpl]()


cdef class BiPartitionSampling(PartitionSampling):
    """
    A wrapper for the C++ class `BiPartitionSampling`.
    """

    def __cinit__(self, float32 holdout_set_size):
        """
        :param holdout_set_size: The fraction of examples to be included in the holdout set (e.g. a value of 0.6
                                 corresponds to 60 % of the available examples). Must be in (0, 1)
        """
        self.partition_sampling_ptr = <shared_ptr[IPartitionSampling]>make_shared[BiPartitionSamplingImpl](
            holdout_set_size)
