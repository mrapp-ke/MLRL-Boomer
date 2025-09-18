from mlrl.common.cython._types cimport uint32


cdef extern from "mlrl/boosting/statistics/quantization_stochastic.hpp" namespace "boosting" nogil:

    cdef cppclass IStochasticQuantizationConfig:

        # Functions:

        uint32 getNumBits() const

        IStochasticQuantizationConfig& setNumBits(uint32 numBits) except +


cdef class StochasticQuantizationConfig:

    # Attributes:

    cdef IStochasticQuantizationConfig* config_ptr
