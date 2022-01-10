from mlrl.common.cython._types cimport uint32


cdef extern from "common/sampling/label_sampling_without_replacement.hpp" nogil:

    cdef cppclass LabelSamplingWithoutReplacementConfigImpl"LabelSamplingWithoutReplacementConfig":

        # Functions:

        uint32 getNumSamples() const

        LabelSamplingWithoutReplacementConfigImpl& setNumSamples(uint32 numSamples) except +


cdef class LabelSamplingWithoutReplacementConfig:

    # Attributes:

    cdef LabelSamplingWithoutReplacementConfigImpl* config_ptr
