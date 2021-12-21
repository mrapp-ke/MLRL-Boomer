from mlrl.common.cython._types cimport uint32, float64

from libcpp.memory cimport unique_ptr


cdef extern from "seco/rule_evaluation/lift_function.hpp" namespace "seco" nogil:

    cdef cppclass ILiftFunction:
        pass


cdef extern from "seco/rule_evaluation/lift_function_peak.hpp" namespace "seco" nogil:

    cdef cppclass PeakLiftFunctionImpl"seco::PeakLiftFunction"(ILiftFunction):

        # Constructors:

        PeakLiftFunctionImpl(uint32 numLabels, uint32 peakLabel, float64 maxLift, float64 curvature) except +


cdef class LiftFunction:

    # Attributes:

    cdef unique_ptr[ILiftFunction] lift_function_ptr


cdef class PeakLiftFunction(LiftFunction):
    pass
