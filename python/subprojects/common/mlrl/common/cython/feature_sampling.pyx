"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._validation import assert_greater_or_equal, assert_less


cdef class FeatureSamplingWithoutReplacementConfig:
    """
    A wrapper for the C++ class `FeatureSamplingWithoutReplacementConfig`.
    """

    def set_sample_size(self, sample_size: float) -> FeatureSamplingWithoutReplacementConfig:
        """
        Sets the fraction of features that should be included in a sample.

        :param sample_size: The fraction of features that should be included in a sample, e.g., a value of 0.6
                            corresponds to 60 % of the available features. Must be in (0, 1) or 0, if the default sample
                            size `floor(log2(numFeatures - 1) + 1)` should be used
        :return:            A `FeatureSamplingWithoutReplacementConfig` that allows further configuration of the method
                            for sampling features
        """
        assert_greater_or_equal('sample_size', sample_size, 0)
        assert_less('sample_size', sample_size, 1)
        self.config_ptr.setSampleSize(sample_size)
        return self
