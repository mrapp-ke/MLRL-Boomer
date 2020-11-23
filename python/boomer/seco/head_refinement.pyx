"""
@author Jakob Steeg (jakob.steeg@gmail.com)
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for C++ classes that allow to find the best head for rules.
"""
from boomer.seco.lift_functions cimport LiftFunction

from libcpp.memory cimport make_shared


cdef class PartialHeadRefinementFactory(HeadRefinementFactory):
    """
    A wrapper for the C++ class `PartialHeadRefinementFactory`.
    """

    def __cinit__(self, LiftFunction lift_function):
        self.head_refinement_factory_ptr = <shared_ptr[IHeadRefinementFactory]>make_shared[PartialHeadRefinementFactoryImpl](
            lift_function.lift_function_ptr)
