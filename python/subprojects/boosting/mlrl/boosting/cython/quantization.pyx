"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.util.validation import assert_greater, assert_less_or_equal


cdef class StochasticQuantizationConfig:
    """
    Allows to configure a method for quantizing statistics that uses a stochastic rounding strategy.
    """

    def get_num_bins(self) -> int:
        """
        Returns the number of bins that are used used for quantized statistics.

        :return: The number of bins that are used
        """
        return self.config_ptr.getNumBins()

    def set_num_bins(self, num_bins: int) -> StochasticQuantizationConfig:
        """
        Sets the number of bins to be used for quantized statistics.

        :param num_bins:    The number of bins to be used. Must be in [0, 128]
        :return:            An `StochasticQuantizationConfig` that allows further configuration of the quantization
                            method
        """
        assert_greater('num_bins', num_bins, 0)
        assert_less_or_equal('num_bins', num_bins, 128)
        self.config_ptr.setNumBins(num_bins)
        return self
