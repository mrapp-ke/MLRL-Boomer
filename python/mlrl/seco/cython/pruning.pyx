"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""

from libcpp.memory cimport make_shared
from mlrl.seco.cython.head_refinement cimport LiftFunction
from mlrl.common.cython.pruning import Pruning

cdef class SecoPruning(Pruning):
    """
    A wrapper for the C++ class `SecoPruning`.
    """

    def __cinit__(self, LiftFunction lift_function):
        self.pruning_ptr = <shared_ptr[IPruning]>make_shared[SecoPruningImpl](lift_function.lift_function_ptr)
