from mlrl.common.cython._types cimport uint32, float32
from libcpp.memory cimport shared_ptr


cdef extern from "boosting/binning/label_binning.hpp" namespace "boosting" nogil:

    cdef cppclass ILabelBinningFactory:
        pass


cdef extern from "boosting/binning/label_binning_equal_width.hpp" namespace "boosting" nogil:

    cdef cppclass EqualWidthLabelBinningFactoryImpl"boosting::EqualWidthLabelBinningFactory"(ILabelBinningFactory):

        # Constructors

        EqualWidthLabelBinningFactoryImpl(float32 binRatio, uint32 minBins, uint32 maxBins) except +


cdef class LabelBinningFactory:

    # Attributes:

    cdef shared_ptr[ILabelBinningFactory] label_binning_factory_ptr


cdef class EqualWidthLabelBinningFactory(LabelBinningFactory):
    pass
