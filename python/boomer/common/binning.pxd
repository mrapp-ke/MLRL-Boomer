from boomer.common._types cimport float32

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/binning/feature_binning.h" nogil:

    cdef cppclass IFeatureBinning:
        pass


cdef extern from "cpp/binning/feature_binning_equal_frequency.h" nogil:

    cdef cppclass EqualFrequencyBinningImpl"EqualFrequencyBinning"(IFeatureBinning):

        # Constructors:

        EqualFrequencyBinningImpl(float32 binRatio) except +


cdef extern from "cpp/binning/feature_binning_equal_width.h" nogil:

    cdef cppclass EqualWidthBinningImpl"EqualWidthBinning"(IFeatureBinning):

        # Constructors:

        EqualWidthBinningImpl(float32 binRatio) except +


cdef class Binning:

    # Attributes:

    cdef shared_ptr[IFeatureBinning] binning_ptr


cdef class EqualFrequencyBinning(Binning):
    pass


cdef class EqualWidthBinning(Binning):
    pass
