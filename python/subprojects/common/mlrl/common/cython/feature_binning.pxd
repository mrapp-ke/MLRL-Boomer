from mlrl.common.cython._types cimport uint32, float32

from libcpp.memory cimport unique_ptr


cdef extern from "common/binning/feature_binning.hpp" nogil:

    cdef cppclass IFeatureBinningFactory:
        pass


cdef extern from "common/binning/feature_binning_equal_frequency.hpp" nogil:

    cdef cppclass EqualFrequencyFeatureBinningFactoryImpl"EqualFrequencyFeatureBinningFactory"(IFeatureBinningFactory):

        # Constructors:

        EqualFrequencyFeatureBinningFactoryImpl(float32 binRatio, uint32 minBins, uint32 maxBins) except +


cdef extern from "common/binning/feature_binning_equal_width.hpp" nogil:

    cdef cppclass EqualWidthFeatureBinningFactoryImpl"EqualWidthFeatureBinningFactory"(IFeatureBinningFactory):

        # Constructors:

        EqualWidthFeatureBinningFactoryImpl(float32 binRatio, uint32 minBins, uint32 maxBins) except +


cdef class FeatureBinningFactory:

    # Attributes:

    cdef unique_ptr[IFeatureBinningFactory] feature_binning_factory_ptr


cdef class EqualFrequencyFeatureBinningFactory(FeatureBinningFactory):
    pass


cdef class EqualWidthFeatureBinningFactory(FeatureBinningFactory):
    pass
