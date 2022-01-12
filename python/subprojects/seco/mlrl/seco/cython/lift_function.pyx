"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._validation import assert_greater, assert_greater_or_equal


cdef class PeakLiftFunctionConfig:
    """
    A wrapper for the C++ class `PeakLiftFunctionConfig`.
    """

    def set_num_labels(self, num_labels: int) -> PeakLiftFunctionConfig:
        """
        Sets the total number of available labels.

        :param num_labels:  The total number of available labels. Must be greater than 0
        :return:            A `PeakLiftFunctionConfig` that allows further configuration of the lift function
        """
        assert_greater('num_labels', num_labels, 0)
        self.config_ptr.setNumLabels(num_labels)
        return self

    def set_peak_label(self, peak_label: int) -> PeakLiftFunctionConfig:
        """
        Sets the index of the label for which the lift should be maximal.

        :param peak_label:  The index of the label for which the lift should be maximal. Must be at least 0
        :return:            A `PeakLiftFunctionConfig` that allows further configuration of the lift function
        """
        assert_greater_or_equal('peak_label', peak_label, 0)
        self.config_ptr.setPeakLabel(peak_label)
        return self

    def set_max_lift(self, max_lift: float) -> PeakLiftFunctionConfig:
        """
        Sets the lift at the peak label.

        :param max_lift:    The lift at the peak label. Must be at least 1
        :return:            A `PeakLiftFunctionConfig` that allows further configuration of the lift function
        """
        assert_greater_or_equal('max_lift', max_lift, 1)
        self.config_ptr.setMaxLift(max_lift)
        return self

    def set_curvature(self, curvature: float) -> PeakLiftFunctionConfig:
        """
        Sets the curvature of the lift function.

        :param curvature:   The curvature of the lift function. A greater value results in a steeper curvature, a
                            smaller value results in a flatter curvature. Must be greater than 0
        :return:            A `PeakLiftFunctionConfig` that allows further configuration of the lift function
        """
        assert_greater('curvature', curvature, 0)
        self.config_ptr.setCurvature(curvature)
        return self
