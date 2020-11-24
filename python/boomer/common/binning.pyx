from libcpp.memory cimport make_shared


cdef class Binning:
    """
    A wrapper for the pure virtual C++ class `IFeatureBinning`.
    """
    pass

cdef class EqualFrequencyFeatureBinning(Binning):
    """
    A wrapper for the C++ class `EqualFrequencyFeatureBinning`.
    """

    def __cinit__(self, float32 bin_ratio):
        self.binning_ptr = <shared_ptr[IFeatureBinning]>make_shared[EqualFrequencyFeatureBinningImpl](bin_ratio)


cdef class EqualWidthFeatureBinning(Binning):
    """
    A wrapper for the C++ class `EqualWidthFeatureBinning`.
    """

    def __cinit__(self, float32 bin_ratio):
        self.binning_ptr = <shared_ptr[IFeatureBinning]>make_shared[EqualWidthFeatureBinningImpl](bin_ratio)
