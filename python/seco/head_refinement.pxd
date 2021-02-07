from common._types cimport uint32, float64
from common.head_refinement cimport HeadRefinementFactory, IHeadRefinementFactory

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/head_refinement/lift_function.hpp" namespace "seco" nogil:

    cdef cppclass ILiftFunction:
        pass


cdef extern from "cpp/head_refinement/lift_function_peak.hpp" namespace "seco" nogil:

    cdef cppclass PeakLiftFunctionImpl"seco::PeakLiftFunction"(ILiftFunction):

        # Constructors:

        PeakLiftFunctionImpl(uint32 numLabels, uint32 peakLabel, float64 maxLift, float64 curvature) except +


cdef extern from "cpp/head_refinement/head_refinement_partial.hpp" namespace "seco" nogil:

    cdef cppclass PartialHeadRefinementFactoryImpl"seco::PartialHeadRefinementFactory"(IHeadRefinementFactory):

        # Constructors:

        PartialHeadRefinementFactoryImpl(shared_ptr[ILiftFunction] liftFunctionPtr) except +


cdef class LiftFunction:

    # Attributes:

    cdef shared_ptr[ILiftFunction] lift_function_ptr


cdef class PeakLiftFunction(LiftFunction):
    pass


cdef class PartialHeadRefinementFactory(HeadRefinementFactory):
    pass
