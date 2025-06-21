"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.util.validation import assert_greater, assert_greater_or_equal


cdef class PeakLiftFunctionConfig:
    """
    Allows to configure a lift function that monotonously increases until a certain number of labels, where the maximum
    lift is reached, and monotonously decreases afterwards.
    """

    def get_peak_label(self) -> int:
        """
        Returns the number of labels for which the lift is maximal.

        :return: The number of labels for which the lift is maximal or 0, if the average label cardinality is used
        """
        return self.config_ptr.getPeakLabel()

    def set_peak_label(self, peak_label: int) -> PeakLiftFunctionConfig:
        """
        Sets the number of labels for which the lift should be maximal.

        :param peak_label:  The number of labels for which the lift should be maximal. Must be at least 1 or 0, if the
                            average label cardinality should be used
        :return:            A `PeakLiftFunctionConfig` that allows further configuration of the lift function
        """
        if peak_label != 0:
            assert_greater_or_equal('peak_label', peak_label, 1)
        self.config_ptr.setPeakLabel(peak_label)
        return self

    def get_max_lift(self) -> float:
        """
        Returns the lift at the peak label.

        :return: The lift at the peak label
        """
        return self.config_ptr.getMaxLift()

    def set_max_lift(self, max_lift: float) -> PeakLiftFunctionConfig:
        """
        Sets the lift at the peak label.

        :param max_lift:    The lift at the peak label. Must be at least 1
        :return:            A `PeakLiftFunctionConfig` that allows further configuration of the lift function
        """
        assert_greater_or_equal('max_lift', max_lift, 1)
        self.config_ptr.setMaxLift(max_lift)
        return self

    def get_curvature(self) -> float:
        """
        Returns the curvature of the lift function.

        :return: The curvature of the lift function
        """
        return self.config_ptr.getCurvature()

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


cdef class KlnLiftFunctionConfig:
    """
    Allows to configure a lift function that monotonously increases according to the natural logarithm of the number of
    labels for which a rule predicts.
    """

    def get_k(self) -> float:
        """
        Returns the value of the parameter "k" that affects the steepness of the lift function.

        :return: The value of the parameter "k"
        """
        return self.config_ptr.getK()

    def set_k(self, k: float) -> KlnLiftFunctionConfig:
        """
        Sets the value of the parameter "k", which affects the steepness of the lift function.

        :param k:   The value of the parameter "k". The steepness of the lift function increases with larger values for
                    "k". Must be greater than 0
        :return:     A `KlnLiftFunctionConfig` that allows further configuration of the lift function
        """
        assert_greater('k', k, 0)
        self.config_ptr.setK(k)
        return self
