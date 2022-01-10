from mlrl.common.cython._types cimport float32


cdef extern from "common/sampling/partition_sampling_bi_stratified_example_wise.hpp" nogil:

    cdef cppclass ExampleWiseStratifiedBiPartitionSamplingConfigImpl"ExampleWiseStratifiedBiPartitionSamplingConfig":

        # Attributes:

        float32 getHoldoutSetSize() const

        ExampleWiseStratifiedBiPartitionSamplingConfigImpl& setHoldoutSetSize(float32 holdoutSetSize) except +


cdef extern from "common/sampling/partition_sampling_bi_stratified_label_wise.hpp" nogil:

    cdef cppclass LabelWiseStratifiedBiPartitionSamplingConfigImpl"LabelWiseStratifiedBiPartitionSamplingConfig":

        # Attributes:

        float32 getHoldoutSetSize() const

        LabelWiseStratifiedBiPartitionSamplingConfigImpl& setHoldoutSetSize(float32 holdoutSetSize) except +


cdef extern from "common/sampling/partition_sampling_bi_random.hpp" nogil:

    cdef cppclass RandomBiPartitionSamplingConfigImpl"RandomBiPartitionSamplingConfig":

        # Attributes:

        float32 getHoldoutSetSize() const

        RandomBiPartitionSamplingConfigImpl& setHoldoutSetSize(float32 holdoutSetSize) except +


cdef class ExampleWiseStratifiedBiPartitionSamplingConfig:

    # Attributes:

    cdef ExampleWiseStratifiedBiPartitionSamplingConfigImpl* config_ptr


cdef class LabelWiseStratifiedBiPartitionSamplingConfig:

    # Attributes:

    cdef LabelWiseStratifiedBiPartitionSamplingConfigImpl* config_ptr


cdef class RandomBiPartitionSamplingConfig:

    # Attributes:

    cdef RandomBiPartitionSamplingConfigImpl* config_ptr
