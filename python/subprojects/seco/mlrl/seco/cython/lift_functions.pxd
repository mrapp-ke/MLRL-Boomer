from mlrl.common.cython._types cimport uint32, float64

from libcpp.memory cimport unique_ptr


cdef extern from "seco/rule_evaluation/lift_function.hpp" namespace "seco" nogil:

    cdef cppclass ILiftFunctionFactory:
        pass


cdef extern from "seco/rule_evaluation/lift_function_peak.hpp" namespace "seco" nogil:

    cdef cppclass PeakLiftFunctionFactoryImpl"seco::PeakLiftFunctionFactory"(ILiftFunctionFactory):

        # Constructors:

        PeakLiftFunctionFactoryImpl(uint32 numLabels, uint32 peakLabel, float64 maxLift, float64 curvature) except +


cdef class LiftFunctionFactory:

    # Attributes:

    cdef unique_ptr[ILiftFunctionFactory] lift_function_factory_ptr


cdef class PeakLiftFunctionFactory(LiftFunctionFactory):
    pass
