"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrapper for C++ classes that implement strategies for pruning classification rules.
"""
from libcpp.memory cimport make_shared


cdef class Pruning:
    """
    A wrapper for the pure virtual C++ class `IPruning`.
    """
    pass


cdef class NoPruning(Pruning):
    """
    A wrapper for the C++ class `NoPruning`.
    """

    def __cinit__(self):
        self.pruning_ptr = <shared_ptr[IPruning]>make_shared[NoPruningImpl]()


cdef class IREP(Pruning):
    """
    A wrapper for the C++ class `IREP`.
    """

    def __cinit__(self):
        self.pruning_ptr = <shared_ptr[IPruning]>make_shared[IREPImpl]()
