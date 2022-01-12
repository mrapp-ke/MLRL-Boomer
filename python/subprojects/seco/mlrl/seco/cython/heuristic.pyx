"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._validation import assert_greater_or_equal


cdef class AccuracyConfig:
    """
    A wrapper for the C++ class `AccuracyConfig`.
    """
    pass


cdef class FMeasureConfig:
    """
    A wrapper for the C++ class `FMeasureConfig`.
    """

    def set_beta(self, beta: float) -> FMeasureConfig:
        """
        Sets the value of the "beta" parameter.

        :param beta:    The value of the "beta" parameter. Must be at least 0
        :return:        A `FMeasureConfig` that allows further configuration of the heuristic
        """
        assert_greater_or_equal('beta', beta, 0)
        self.config_ptr.setBeta(beta)
        return self


cdef class LaplaceConfig:
    """
    A wrapper for the C++ class `LaplaceConfig`.
    """
    pass


cdef class MEstimateConfig:
    """
    A wrapper for the C++ class `MEstimateConfig`.
    """

    def set_m(self, m: float) -> MEstimateConfig:
        """
        Sets the value of the "m" parameter.

        :param m:   The value of the "m" parameter. Must be at least 0
        :return:    A `MEstimateConfig` that allows further configuration of the heuristic
        """
        assert_greater_or_equal('m', m, 0)
        self.config_ptr.setM(m)
        return self


cdef class PrecisionConfig:
    """
    A wrapper for the C++ class `PrecisionConfig`.
    """
    pass


cdef class RecallConfig:
    """
    A wrapper for the C++ class `RecallConfig`.
    """
    pass


cdef class WraConfig:
    """
    A wrapper for the C++ class `WraConfig`.
    """
    pass
