from libcpp cimport bool

from mlrl.common.cython._types cimport float32, uint32


cdef extern from "mlrl/common/random/rng.hpp" nogil:

    cdef cppclass IRNGConfig"RNGConfig":

        # Functions:

        IRNGConfig& setRandomState(uint32 randomState) except +

        uint32 getRandomState() const


cdef class RNGConfig:

    # Attributes:

    cdef IRNGConfig* config_ptr
