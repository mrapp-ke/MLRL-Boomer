from boomer.common.head_refinement cimport HeadRefinementFactory, IHeadRefinementFactory
from boomer.seco.lift_functions cimport ILiftFunction

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/head_refinement/head_refinement_partial.h" namespace "seco" nogil:

    cdef cppclass PartialHeadRefinementFactoryImpl"seco::PartialHeadRefinementFactory"(IHeadRefinementFactory):

        # Constructors:

        PartialHeadRefinementFactoryImpl(shared_ptr[ILiftFunction] liftFunctionPtr) except +


cdef class PartialHeadRefinementFactory(HeadRefinementFactory):
    pass
