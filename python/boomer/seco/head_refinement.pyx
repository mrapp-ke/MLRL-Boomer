from boomer.seco.lift_functions cimport LiftFunction

from libcpp.memory cimport make_shared


cdef class PartialHeadRefinementFactory(HeadRefinementFactory):
    """
    A wrapper for the C++ class `PartialHeadRefinementFactoryImpl`.
    """

    def __cinit__(self, LiftFunction lift_function):
        self.head_refinement_factory_ptr = <shared_ptr[IHeadRefinementFactory]>make_shared[PartialHeadRefinementFactoryImpl](
            lift_function.lift_function_ptr)
