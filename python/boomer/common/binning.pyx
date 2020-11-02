from libcpp.memory cimport make_shared


cdef class Binning:
    """
    A wrapper for the pure virtual C++ class `IBinning`.
    """
    pass

cdef class EqualFrequencyBinning(Binning):
    def __cinit__(self, float32 bin_ratio):
        self.binning_ptr = <shared_ptr[IBinning]>make_shared[EqualFrequencyBinningImpl](bin_ratio)

cdef class EqualWidthBinning(Binning):
    def __cinit__(self, float32 bin_ratio):
        self.binning_ptr = <shared_ptr[IBinning]>make_shared[EqualWidthBinningImpl](bin_ratio)