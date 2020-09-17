from boomer.common.head_refinement cimport HeadRefinement, AbstractHeadRefinement
from boomer.seco.lift_functions cimport AbstractLiftFunction

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/head_refinement.h" namespace "seco" nogil:

    cdef cppclass PartialHeadRefinementImpl(AbstractHeadRefinement):

        # Constructors:

        PartialHeadRefinementImpl(shared_ptr[AbstractLiftFunction] liftFunctionPtr) except +


cdef class PartialHeadRefinement(HeadRefinement):
    pass
