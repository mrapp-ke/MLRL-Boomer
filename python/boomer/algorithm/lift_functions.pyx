from libc.math cimport pow

cdef class LiftFunction:
    """
    Base class for all relaxation lift functions.
    """

    cdef float64 eval(self, intp label_count):
        """
        Calculates and returns the relaxation lift for a certain number of labels
        :param label_count: The number of labels for which the relaxation lift should be calculated
        :return: The calculated relaxation lift
        """
        pass

    cdef float64 get_maximum_lift(self):
        """
        :return: The relaxation lift at the peak label
        """
        pass

cdef class PeakLiftFunction(LiftFunction):
    """
    Lift function which monotonously increases until the point with the highest lift is reached and monotonously
    decreases after that point.
    """

    def __cinit__(self, intp maximum_number_of_labels, float64 maximum, float64 maximum_lift, float64 curvature):
        """
        :param maximum_number_of_labels: The maximum number of labels predictable in the set
        :param maximum: The number of labels for which the relaxation lift is maximal
        :param maximum_lift: The relaxation lift at the peak label
        :param curvature: The curvature of the lift function (a higher value means a steeper curvature and a lower value
            means a flatter curvature
        """
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