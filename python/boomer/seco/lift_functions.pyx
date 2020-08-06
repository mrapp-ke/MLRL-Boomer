"""
Provides Cython wrappers for C++ classes that implement different lift functions.

@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""


cdef class LiftFunction:
    """
    A wrapper for the C++ class `AbstractLiftFunction`.
    """

    cdef float64 calculate_lift(self, intp num_labels) nogil:
        """
        Calculates and returns the lift for a specific number of labels.

        :param num_labels:  The number of labels for which the lift should be calculated
        :return:            The lift that has been calculated
        """
        cdef AbstractLiftFunction* lift_function = self.lift_function
        return lift_function.calculateLift(num_labels)

    cdef float64 get_max_lift(self) nogil:
        """
        Returns the maximum lift possible.

        :return:    The maximum lift possible
        """
        cdef AbstractLiftFunction* lift_function = self.lift_function
        return lift_function.getMaxLift()


cdef class PeakLiftFunction(LiftFunction):
    """
    A wrapper for the C++ class `PeakLiftFunctionImpl`.
    """

    def __cinit__(self, intp num_labels, intp peak_label, float64 max_lift, float64 curvature):
        """
        :param num_labels:  The total number of available labels. Must be greater than 0
        :param peak_label:  The number of labels for which the lift is maximum. Must be in [1, numLabels]
        :param max_lift:    The lift at the peak label. Must be at least 1
        :param curvature:   The curvature of the lift function. A greater value results in a steeper curvature, a
                            smaller value results in a flatter curvature. Must be greater than 0
        """
        self.lift_function = new PeakLiftFunctionImpl(num_labels, peak_label, max_lift, curvature)

    def __dealloc__(self):
        del self.lift_function
