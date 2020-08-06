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

    cdef float64 eval(self, intp label_count);

    cdef float64 get_max_lift(self);


cdef class PeakLiftFunction(LiftFunction):

    # Attributes:

    cdef intp num_labels

    cdef intp peak_label

    cdef float64 max_lift

    cdef float64 exponent

    # Functions:

    cdef float64 eval(self, intp label_count);

    cdef float64 get_max_lift(self);
