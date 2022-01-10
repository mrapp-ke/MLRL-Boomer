from mlrl.common.cython._types cimport float32


cdef extern from "common/sampling/feature_sampling_without_replacement.hpp" nogil:

    cdef cppclass FeatureSamplingWithoutReplacementConfigImpl"FeatureSamplingWithoutReplacementConfig":

        # Functions:

        float32 getSampleSize() const

        FeatureSamplingWithoutReplacementConfigImpl& setSampleSize(float32 sampleSize) except +


cdef class FeatureSamplingWithoutReplacementConfig:

    # Attributes:

    cdef FeatureSamplingWithoutReplacementConfigImpl* config_ptr
