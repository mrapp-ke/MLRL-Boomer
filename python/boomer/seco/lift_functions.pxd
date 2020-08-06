from boomer.common._arrays cimport intp, float64


cdef extern from "cpp/lift_functions.h" namespace "seco" nogil:

    cdef cppclass AbstractLiftFunction:

        # Functions:

        float64 calculateLift(intp numLabels)

        float64 getMaxLift()


    cdef cppclass PeakLiftFunctionImpl(AbstractLiftFunction):

        # Constructors:

        PeakLiftFunctionImpl(intp numLabels, intp peakLabel, float64 maxLift, float64 curvature) except +

        # Functions:

        float64 calculateLift(intp numLabels)

        float64 getMaxLift()


cdef class LiftFunction:

    # Attributes:

    cdef AbstractLiftFunction* lift_function


cdef class PeakLiftFunction(LiftFunction):
    pass
