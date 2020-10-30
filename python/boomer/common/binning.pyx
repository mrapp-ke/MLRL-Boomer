

cdef class Binning:
    """
    A wrapper for the pure virtual C++ class `IBinning`.
    """
    pass

cdef class EqualFrequencyBinning:
    def __cinit__(self):
        pass

cdef class EqualWidthBinning:
    def __cinit__(self):
        pass