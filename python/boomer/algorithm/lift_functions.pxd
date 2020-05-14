from boomer.algorithm._arrays cimport intp, float64

cdef class LiftFunction:

    cdef float64 eval(self, intp label_count);

    cdef float64 get_maximum_lift(self);

cdef class PeakLiftFunction(LiftFunction):

    # Attributes:

    cdef float64 maximum_labels

    cdef float64 maximum

    cdef float64 maximum_lift

    cdef float64 curvature

    # Functions:

    cdef float64 eval(self, intp label_count);

    cdef float64 get_maximum_lift(self);

    cdef float64 eval_before_maximum(self, intp label_count);

    cdef float64 eval_after_maximum(self, intp label_count);