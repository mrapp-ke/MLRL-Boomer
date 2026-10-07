from mlrl.common.cython._types cimport float32, uint32


cdef extern from "mlrl/common/sampling/output_sampling_without_replacement.hpp" nogil:

    cdef cppclass IOutputSamplingWithoutReplacementConfig:

        # Functions:

        float32 getSampleSize() const

        IOutputSamplingWithoutReplacementConfig& setSampleSize(float32 sampleSize)

        uint32 getMinSamples() const

        IOutputSamplingWithoutReplacementConfig& setMinSamples(uint32 minSamples)

        uint32 getMaxSamples() const

        IOutputSamplingWithoutReplacementConfig& setMaxSamples(uint32 maxSamples)


cdef class OutputSamplingWithoutReplacementConfig:

    # Attributes:

    cdef IOutputSamplingWithoutReplacementConfig* config_ptr
