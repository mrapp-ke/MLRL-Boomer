"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.validation import assert_greater_or_equal


cdef class OutputSamplingWithoutReplacementConfig:
    """
    Allows to configure a method for sampling outputs without replacement.
    """

    def get_num_samples(self) -> int:
        """
        Returns the number of outputs that are included in a sample.

        :return: The number of outputs that are included in a sample
        """
        return self.config_ptr.getNumSamples()

    def set_num_samples(self, num_samples: int) -> OutputSamplingWithoutReplacementConfig:
        """
        Sets the number of outputs that should be included in a sample.

        :param num_samples: The number of outputs that should be included in a sample. Must be at least 1
        :return:            An `OutputSamplingWithoutReplacementConfig` that allows further configuration of the
                            sampling method
        """
        assert_greater_or_equal('num_samples', num_samples, 1)
        self.config_ptr.setNumSamples(num_samples)
        return self
