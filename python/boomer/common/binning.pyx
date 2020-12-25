from libcpp.memory cimport make_shared


cdef class FeatureBinning:
    """
    A wrapper for the pure virtual C++ class `IFeatureBinning`.
    """
    pass


cdef class EqualFrequencyFeatureBinning(FeatureBinning):
    """
    A wrapper for the C++ class `EqualFrequencyFeatureBinning`.
    """

    def __cinit__(self, float32 bin_ratio, uint32 min_bins, uint32 max_bins):
        self.binning_ptr = <shared_ptr[IFeatureBinning]>make_shared[EqualFrequencyFeatureBinningImpl](bin_ratio,
                                                                                                      min_bins,
                                                                                                      max_bins)


cdef class EqualWidthFeatureBinning(FeatureBinning):
    """
    A wrapper for the C++ class `EqualWidthFeatureBinning`.
    """

    def __cinit__(self, float32 bin_ratio, uint32 min_bins, uint32 max_bins):
        self.binning_ptr = <shared_ptr[IFeatureBinning]>make_shared[EqualWidthFeatureBinningImpl](bin_ratio, min_bins,
                                                                                                  max_bins)
