from mlrl.common.cython._types cimport float32, uint32


cdef extern from "mlrl/common/sampling/instance_sampling_stratified_example_wise.hpp" nogil:

    cdef cppclass IExampleWiseStratifiedInstanceSamplingConfig:

        # Functions:

        float32 getSampleSize() const

        IExampleWiseStratifiedInstanceSamplingConfig& setSampleSize(float32 sampleSize)


cdef extern from "mlrl/common/sampling/instance_sampling_stratified_output_wise.hpp" nogil:

    cdef cppclass IOutputWiseStratifiedInstanceSamplingConfig:

        # Functions:

        float32 getSampleSize() const

        IOutputWiseStratifiedInstanceSamplingConfig& setSampleSize(float32 sampleSize)


cdef extern from "mlrl/common/sampling/instance_sampling_with_replacement.hpp" nogil:

    cdef cppclass IInstanceSamplingWithReplacementConfig:

        # Functions:

        float32 getSampleSize() const

        IInstanceSamplingWithReplacementConfig& setSampleSize(float32 sampleSize)

        uint32 getMinSamples() const

        IExampleWiseStratifiedInstanceSamplingConfig& setMinSamples(float32 minSamples)

        uint32 getMaxSamples() const

        IExampleWiseStratifiedInstanceSamplingConfig& setMaxSamples(float32 maxSamples)


cdef extern from "mlrl/common/sampling/instance_sampling_without_replacement.hpp" nogil:

    cdef cppclass IInstanceSamplingWithoutReplacementConfig:

        # Functions:

        float32 getSampleSize() const

        IInstanceSamplingWithoutReplacementConfig& setSampleSize(float32 sampleSize)


cdef class ExampleWiseStratifiedInstanceSamplingConfig:

    # Attributes:

    cdef IExampleWiseStratifiedInstanceSamplingConfig* config_ptr


cdef class OutputWiseStratifiedInstanceSamplingConfig:

    # Attributes:

    cdef IOutputWiseStratifiedInstanceSamplingConfig* config_ptr


cdef class InstanceSamplingWithReplacementConfig:

    # Attributes:

    cdef IInstanceSamplingWithReplacementConfig* config_ptr


cdef class InstanceSamplingWithoutReplacementConfig:

    # Attributes:

    cdef IInstanceSamplingWithoutReplacementConfig* config_ptr
