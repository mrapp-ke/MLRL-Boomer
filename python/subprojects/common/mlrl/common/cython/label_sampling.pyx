"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.validation import assert_greater_or_equal


cdef class LabelSamplingWithoutReplacementConfig:
    """
    A wrapper for the C++ class `FeatureSamplingWithoutReplacementConfig`.
    """

    def set_num_samples(self, num_samples: int) -> LabelSamplingWithoutReplacementConfig:
        """
        Sets the number of labels that should be included in a sample.

        :param num_samples: The number of labels that should be included in a sample. Must be at least 1
        :return:            A `LabelSamplingWithoutReplacementConfig` that allows further configuration of the strategy
                            for sampling labels
        """
        assert_greater_or_equal('num_samples', num_samples, 1)
        self.config_ptr.setNumSamples(num_samples)
        return self
