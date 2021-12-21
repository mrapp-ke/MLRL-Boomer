from libcpp.memory cimport unique_ptr


cdef extern from "common/pruning/pruning.hpp" nogil:

    cdef cppclass IPruningFactory:
        pass


cdef extern from "common/pruning/pruning_no.hpp" nogil:

    cdef cppclass NoPruningFactoryImpl"NoPruningFactory"(IPruningFactory):
        pass


cdef extern from "common/pruning/pruning_irep.hpp" nogil:

    cdef cppclass IrepFactoryImpl"IrepFactory"(IPruningFactory):
        pass


cdef class PruningFactory:

    # Attributes:

    cdef unique_ptr[IPruningFactory] pruning_factory_ptr


cdef class NoPruningFactory(PruningFactory):
    pass


cdef class IrepFactory(PruningFactory):
    pass
