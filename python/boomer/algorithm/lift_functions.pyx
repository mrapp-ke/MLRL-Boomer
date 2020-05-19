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

    def __cinit__(self, intp num_labels, float64 peak_label, float64 max_lift, float64 curvature):
        """
        :param num_labels: The maximum number of labels predictable in the set. Must be more than zero
        :param peak_label: The number of labels for which the relaxation lift is maximal. Must be in [1,num_labels]
        :param max_lift: The relaxation lift at the peak label. Must be one or more
        :param curvature: The curvature of the lift function. A higher value means a steeper curvature and a lower value
            means a flatter curvature. Must be more than 0
        """
        self.num_labels = num_labels
        self.peak_label = peak_label
        self.max_lift = max_lift
        self.exponent = 1.0 / curvature

    cdef float64 eval(self, intp label_count):
        cdef float64 normalization
        cdef float64 boost

        if label_count < self.peak_label:
            normalization = (label_count - 1.0) / (self.peak_label - 1.0)
        elif label_count > self.peak_label:
            normalization = (label_count - self.num_labels) / (self.num_labels - self.peak_label)
        else:
            return self.max_lift

        boost = 1.0 + pow(normalization, self.exponent) * (self.max_lift - 1.0)
        return boost

cdef float64 get_maximum_lift(self):
    return self.maximum_lift
