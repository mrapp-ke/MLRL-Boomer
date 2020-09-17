from boomer.seco.lift_functions cimport LiftFunction

from libcpp.memory cimport make_shared


cdef class PartialHeadRefinement(HeadRefinement):
    """
    A wrapper for the C++ class `PartialHeadRefinementImpl`.
    """

    def __cinit__(self, LiftFunction lift_function):
        self.head_refinement_ptr = <shared_ptr[AbstractHeadRefinement]>make_shared[PartialHeadRefinementImpl](
            lift_function.lift_function_ptr)
