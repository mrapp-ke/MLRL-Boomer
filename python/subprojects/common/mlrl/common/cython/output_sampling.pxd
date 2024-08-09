from mlrl.common.cython._types cimport uint32


cdef extern from "mlrl/common/sampling/output_sampling_without_replacement.hpp" nogil:

    cdef cppclass IOutputSamplingWithoutReplacementConfig:

        # Functions:

        uint32 getNumSamples() const

        IOutputSamplingWithoutReplacementConfig& setNumSamples(uint32 numSamples) except +


cdef class OutputSamplingWithoutReplacementConfig:

    # Attributes:

    cdef IOutputSamplingWithoutReplacementConfig* config_ptr
