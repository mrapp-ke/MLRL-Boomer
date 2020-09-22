from boomer.common.head_refinement cimport HeadRefinement, IHeadRefinement
from boomer.seco.lift_functions cimport ILiftFunction

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/head_refinement.h" namespace "seco" nogil:

    cdef cppclass PartialHeadRefinementImpl(IHeadRefinement):

        # Constructors:

        PartialHeadRefinementImpl(shared_ptr[ILiftFunction] liftFunctionPtr) except +


cdef class PartialHeadRefinement(HeadRefinement):
    pass
