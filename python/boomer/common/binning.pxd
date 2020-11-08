from boomer.common._types cimport float32

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/binning.h" nogil:

    cdef cppclass IBinning:
        pass


    cdef cppclass EqualFrequencyBinningImpl(IBinning):

        # Constructors:

        EqualFrequencyBinningImpl(float32 binRatio) except +


    cdef cppclass EqualWidthBinningImpl(IBinning):

        # Constructors:

        EqualWidthBinningImpl(float32 binRatio) except +


cdef class Binning:

    # Attributes:

    cdef shared_ptr[IBinning] binning_ptr


cdef class EqualFrequencyBinning(Binning):
    pass


cdef class EqualWidthBinning(Binning):
    pass
