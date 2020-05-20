from boomer.algorithm._arrays cimport intp, float64

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