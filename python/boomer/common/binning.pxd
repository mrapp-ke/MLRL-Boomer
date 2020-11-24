from boomer.common._types cimport float32

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/binning/feature_binning.h" nogil:

    cdef cppclass IFeatureBinning:
        pass


cdef extern from "cpp/binning/feature_binning_equal_frequency.h" nogil:

    cdef cppclass EqualFrequencyFeatureBinningImpl"EqualFrequencyFeatureBinning"(IFeatureBinning):

        # Constructors:

        EqualFrequencyFeatureBinningImpl(float32 binRatio) except +


cdef extern from "cpp/binning/feature_binning_equal_width.h" nogil:

    cdef cppclass EqualWidthFeatureBinningImpl"EqualWidthFeatureBinning"(IFeatureBinning):

        # Constructors:

        EqualWidthFeatureBinningImpl(float32 binRatio) except +


cdef class FeatureBinning:

    # Attributes:

    cdef shared_ptr[IFeatureBinning] binning_ptr


cdef class EqualFrequencyFeatureBinning(FeatureBinning):
    pass


cdef class EqualWidthFeatureBinning(FeatureBinning):
    pass
