from mlrl.common.cython._types cimport uint32


cdef extern from "common/multi_threading/multi_threading_manual.hpp" nogil:

    cdef cppclass IManualMultiThreadingConfig:

        # Functions:

        uint32 getNumPreferredThreads() const

        IManualMultiThreadingConfig& setNumPreferredThreads(uint32 numPreferredThreads) except +


cdef class ManualMultiThreadingConfig:

    # Attributes:

    cdef IManualMultiThreadingConfig* config_ptr
