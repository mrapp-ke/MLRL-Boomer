from libc.math cimport pow

cdef class LiftFunction:

    cdef float64 eval(self, intp label_count):
        pass

    cdef float64 get_maximum_lift(self):
        pass

cdef class PeakLiftFunction(LiftFunction):

    def __cinit__(self, intp maximum_number_of_labels, float64 maximum, float64 maximum_lift, float64 curvature):
        self.maximum_labels = maximum_number_of_labels
        self.maximum = maximum
        self.maximum_lift = maximum_lift
        self.exponent = 1.0 / curvature

    cdef float64 eval(self, intp label_count):
        if label_count < self.maximum:
            return self.eval_before_maximum(label_count)
        elif label_count > self.maximum:
            return self.eval_after_maximum(label_count)
        return self.maximum_lift

    cdef float64 eval_before_maximum(self, intp label_count):
        cdef float64 normalization = (label_count - 1.0) / (self.maximum - 1)
        cdef float64 boost = 1.0 + pow(normalization, self.exponent) * (self.maximum_lift - 1)
        return boost

    cdef float64 eval_after_maximum(self, intp label_count):
        cdef float64 normalization = (label_count - self.maximum_labels) / (self.maximum_labels - self.maximum )
        cdef float64 boost = 1.0 + pow(normalization, self.exponent) * (self.maximum_lift - 1.0)
        return boost

    cdef float64 get_maximum_lift(self):
        return self.maximum_lift