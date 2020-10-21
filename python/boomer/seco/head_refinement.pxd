from boomer.common.head_refinement cimport HeadRefinementFactory, IHeadRefinementFactory
from boomer.seco.lift_functions cimport ILiftFunction

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/head_refinement.h" namespace "seco" nogil:

    cdef cppclass PartialHeadRefinementFactoryImpl(IHeadRefinementFactory):

        # Constructors:

        PartialHeadRefinementFactoryImpl(shared_ptr[ILiftFunction] liftFunctionPtr) except +


cdef class PartialHeadRefinementFactory(HeadRefinementFactory):
    pass
