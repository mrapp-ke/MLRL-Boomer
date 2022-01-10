cdef extern from "common/pruning/pruning_irep.hpp" nogil:

    cdef cppclass IrepConfigImpl"IrepConfig":
        pass


cdef class IrepConfig:

    cdef IrepConfigImpl config_ptr
