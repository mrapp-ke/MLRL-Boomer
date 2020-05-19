from boomer.algorithm._arrays cimport intp, float64

cdef class LiftFunction:

    cdef float64 eval(self, intp label_count);

    cdef float64 get_maximum_lift(self);

cdef class PeakLiftFunction(LiftFunction):

    # Attributes:

    cdef float64 num_labels

    cdef float64 peak_label

    cdef float64 max_lift

    cdef float64 exponent

    # Functions:

    cdef float64 eval(self, intp label_count);

    cdef float64 get_maximum_lift(self);