from libcpp.memory cimport shared_ptr

from mlrl.seco.cython.head_refinement cimport ILiftFunction
from mlrl.common.cython.pruning cimport IPruning, Pruning


cdef extern from "seco/pruning/pruning_seco.hpp" nogil:

    cdef cppclass SecoPruningImpl"seco::SecoPruning"(IPruning):
        # Constructor:

        SecoIREPImpl(shared_ptr[ILiftFunction] liftFunctionPtr) except +


cdef class SecoPruning(Pruning):
    pass
