from libcpp.memory cimport shared_ptr


cdef extern from "cpp/pruning/pruning.h" nogil:

    cdef cppclass IPruning:
        pass


cdef extern from "cpp/pruning/pruning_no.h" nogil:

    cdef cppclass NoPruningImpl"NoPruning"(IPruning):
        pass


cdef extern from "cpp/pruning/pruning_irep.h" nogil:

    cdef cppclass IREPImpl"IREP"(IPruning):
        pass


cdef class Pruning:

    # Attributes:

    cdef shared_ptr[IPruning] pruning_ptr


cdef class NoPruning(Pruning):
    pass


cdef class IREP(Pruning):
    pass
