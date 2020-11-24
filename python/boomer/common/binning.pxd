from boomer.common._types cimport float32

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/binning/feature_binning.h" nogil:

    cdef cppclass IBinning:
        pass


cdef extern from "cpp/binning/feature_binning_equal_frequency.h" nogil:

    cdef cppclass EqualFrequencyBinningImpl"EqualFrequencyBinning"(IBinning):

        # Constructors:

        EqualFrequencyBinningImpl(float32 binRatio) except +


cdef extern from "cpp/binning/feature_binning_equal_width.h" nogil:

    cdef cppclass EqualWidthBinningImpl"EqualWidthBinning"(IBinning):

        # Constructors:

        EqualWidthBinningImpl(float32 binRatio) except +


cdef class Binning:

    # Attributes:

    cdef shared_ptr[IBinning] binning_ptr


cdef class EqualFrequencyBinning(Binning):
    pass


cdef class EqualWidthBinning(Binning):
    pass
