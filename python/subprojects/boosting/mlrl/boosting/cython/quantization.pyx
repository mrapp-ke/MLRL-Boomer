"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.util.validation import assert_greater


cdef class StochasticQuantizationConfig:
    """
    Allows to configure a method for quantizing statistics that uses a stochastic rounding strategy.
    """

    def get_num_bits(self) -> int:
        """
        Returns the number of bits that are used used for quantized statistics.

        :return: The number of bits that are used
        """
        return self.config_ptr.getNumBits()

    def set_num_bits(self, num_bits: int) -> StochasticQuantizationConfig:
        """
        Sets the number of bits to be used for quantized statistics.

        :param num_bits:    The number of bits to be used. Must be greater than 0
        :return:            An `StochasticQuantizationConfig` that allows further configuration of the quantization
                            method
        """
        assert_greater('num_bits', num_bits, 0)
        self.config_ptr.setNumBits(num_bits)
        return self
