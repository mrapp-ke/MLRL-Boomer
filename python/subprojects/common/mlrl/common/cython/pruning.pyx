"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport make_unique


cdef class PruningFactory:
    """
    A wrapper for the pure virtual C++ class `IPruningFactory`.
    """
    pass


cdef class NoPruningFactory(PruningFactory):
    """
    A wrapper for the C++ class `NoPruningFactory`.
    """

    def __cinit__(self):
        self.pruning_factory_ptr = <unique_ptr[IPruningFactory]>make_unique[NoPruningFactoryImpl]()


cdef class IrepFactory(PruningFactory):
    """
    A wrapper for the C++ class `IrepFactory`.
    """

    def __cinit__(self):
        self.pruning_factory_ptr = <unique_ptr[IPruningFactory]>make_unique[IrepFactoryImpl]()
