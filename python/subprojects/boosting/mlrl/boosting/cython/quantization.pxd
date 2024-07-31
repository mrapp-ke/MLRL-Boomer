from mlrl.common.cython._types cimport uint8


cdef extern from "mlrl/boosting/statistics/quantization_stochastic.hpp" namespace "boosting" nogil:

    cdef cppclass IStochasticQuantizationConfig:

        # Functions:

        uint8 getNumBins() const

        IStochasticQuantizationConfig& setNumBins(uint8 numBins) except +


cdef class StochasticQuantizationConfig:

    # Attributes:

    cdef IStochasticQuantizationConfig* config_ptr
