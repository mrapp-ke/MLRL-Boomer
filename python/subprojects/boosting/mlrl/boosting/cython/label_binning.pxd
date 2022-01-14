from mlrl.common.cython._types cimport uint32, float32


cdef extern from "boosting/binning/label_binning_equal_width.hpp" namespace "boosting" nogil:

    cdef cppclass EqualWidthLabelBinningConfigImpl"boosting::EqualWidthLabelBinningConfig":

        # Functions:

        float32 getBinRatio() const

        EqualWidthLabelBinningConfigImpl& setBinRatio(float32 binRatio) except +

        uint32 getMinBins() const

        EqualWidthLabelBinningConfigImpl& setMinBins(uint32 minBins) except +

        uint32 getMaxBins() const

        EqualWidthLabelBinningConfigImpl& setMaxBins(uint32 maxBins) except +


cdef class EqualWidthLabelBinningConfig:

    # Attributes:

    cdef EqualWidthLabelBinningConfigImpl* config_ptr
