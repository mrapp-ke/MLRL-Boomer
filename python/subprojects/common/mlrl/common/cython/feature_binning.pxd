from mlrl.common.cython._types cimport uint32, float32


cdef extern from "common/binning/feature_binning_equal_width.hpp" nogil:

    cdef cppclass EqualWidthFeatureBinningConfigImpl"EqualWidthFeatureBinningConfig":

        # Functions:

        EqualWidthFeatureBinningConfigImpl& setBinRatio(float32 binRatio) except +

        float32 getBinRatio() const

        EqualWidthFeatureBinningConfigImpl& setMinBins(uint32 minBins) except +

        uint32 getMinBins() const

        EqualWidthFeatureBinningConfigImpl& setMaxBins(uint32 maxBins) except +

        uint32 getMaxBins() const


cdef extern from "common/binning/feature_binning_equal_frequency.hpp" nogil:

    cdef cppclass EqualFrequencyFeatureBinningConfigImpl"EqualFrequencyFeatureBinningConfig":

        # Functions:

        EqualFrequencyFeatureBinningConfigImpl& setBinRatio(float32 binRatio) except +

        float32 getBinRatio() const

        EqualFrequencyFeatureBinningConfigImpl& setMinBins(uint32 minBins) except +

        uint32 getMinBins() const

        EqualFrequencyFeatureBinningConfigImpl& setMaxBins(uint32 maxBins) except +

        uint32 getMaxBins() const


cdef class EqualWidthFeatureBinningConfig:

    # Attributes:

    cdef EqualWidthFeatureBinningConfigImpl* config_ptr


cdef class EqualFrequencyFeatureBinningConfig:

    # Attributes:

    cdef EqualFrequencyFeatureBinningConfigImpl* config_ptr
