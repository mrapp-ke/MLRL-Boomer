cpdef object set_debug_flag()

cdef extern from "common/debugging/debug.hpp":

      cdef void setDebugFlag()