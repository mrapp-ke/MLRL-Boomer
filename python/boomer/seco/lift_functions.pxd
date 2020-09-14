from boomer.common._arrays cimport uint32, float64

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/lift_functions.h" namespace "seco" nogil:

    cdef cppclass AbstractLiftFunction:

        # Functions:

        float64 calculateLift(uint32 numLabels)

        float64 getMaxLift()


    cdef cppclass PeakLiftFunctionImpl(AbstractLiftFunction):

        # Constructors:

        PeakLiftFunctionImpl(uint32 numLabels, uint32 peakLabel, float64 maxLift, float64 curvature) except +


cdef class LiftFunction:

    # Attributes:

    cdef shared_ptr[AbstractLiftFunction] lift_function_ptr


cdef class PeakLiftFunction(LiftFunction):
    pass
