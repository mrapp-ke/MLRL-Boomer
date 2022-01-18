cdef extern from "common/pruning/pruning_irep.hpp" nogil:

    cdef cppclass IIrepConfig:
        pass


cdef class IrepConfig:

    cdef IIrepConfig* config_ptr
