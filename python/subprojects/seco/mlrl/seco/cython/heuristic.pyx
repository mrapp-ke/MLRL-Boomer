"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.util.validation import assert_greater_or_equal


cdef class FMeasureConfig:
    """
    Allows to configure a heuristic that calculates as the (weighted) harmonic mean between the heuristics "Precision"
    and "Recall", where the parameter "beta" allows to trade off between both heuristics. If beta = 1, both heuristics
    are weighed equally. If beta = 0, this heuristic is equivalent to "Precision". As beta approaches infinity, this
    heuristic becomes equivalent to "Recall".
    """

    def get_beta(self) -> float:
        """
        Returns the value of the "beta" parameter.

        :return: The value of the "beta" parameter
        """
        return self.config_ptr.getBeta()

    def set_beta(self, beta: float) -> FMeasureConfig:
        """
        Sets the value of the "beta" parameter.

        :param beta:    The value of the "beta" parameter. Must be at least 0
        :return:        A `FMeasureConfig` that allows further configuration of the heuristic
        """
        assert_greater_or_equal('beta', beta, 0)
        self.config_ptr.setBeta(beta)
        return self


cdef class MEstimateConfig:
    """
    Allows to configure a heuristic that trades off between the heuristics "Precision" and "WRA", where the "m"
    parameter controls the trade-off between both heuristics. If m = 0, this heuristic is equivalent to "Precision". As
    m approaches infinity, the isometrics of this heuristic become equivalent to those of "WRA".
    """

    def get_m(self) -> float:
        """
        Returns the value of the "m" parameter.

        :return: The value of the "m" parameter
        """
        return self.config_ptr.getM()

    def set_m(self, m: float) -> MEstimateConfig:
        """
        Sets the value of the "m" parameter.

        :param m:   The value of the "m" parameter. Must be at least 0
        :return:    A `MEstimateConfig` that allows further configuration of the heuristic
        """
        assert_greater_or_equal('m', m, 0)
        self.config_ptr.setM(m)
        return self
