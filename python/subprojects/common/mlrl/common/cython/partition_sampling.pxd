from mlrl.common.cython._types cimport float32


cdef extern from "mlrl/common/sampling/partition_sampling_bi_stratified_example_wise.hpp" nogil:

    cdef cppclass IExampleWiseStratifiedBiPartitionSamplingConfig:

        # Attributes:

        float32 getHoldoutSetSize() const

        IExampleWiseStratifiedBiPartitionSamplingConfig& setHoldoutSetSize(float32 holdoutSetSize) except +


cdef extern from "mlrl/common/sampling/partition_sampling_bi_stratified_output_wise.hpp" nogil:

    cdef cppclass IOutputWiseStratifiedBiPartitionSamplingConfig:

        # Attributes:

        float32 getHoldoutSetSize() const

        IOutputWiseStratifiedBiPartitionSamplingConfig& setHoldoutSetSize(float32 holdoutSetSize) except +


cdef extern from "mlrl/common/sampling/partition_sampling_bi_random.hpp" nogil:

    cdef cppclass IRandomBiPartitionSamplingConfig:

        # Attributes:

        float32 getHoldoutSetSize() const

        IRandomBiPartitionSamplingConfig& setHoldoutSetSize(float32 holdoutSetSize) except +


cdef class ExampleWiseStratifiedBiPartitionSamplingConfig:

    # Attributes:

    cdef IExampleWiseStratifiedBiPartitionSamplingConfig* config_ptr


cdef class OutputWiseStratifiedBiPartitionSamplingConfig:

    # Attributes:

    cdef IOutputWiseStratifiedBiPartitionSamplingConfig* config_ptr


cdef class RandomBiPartitionSamplingConfig:

    # Attributes:

    cdef IRandomBiPartitionSamplingConfig* config_ptr
