"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.validation import assert_greater, assert_greater_or_equal, assert_less


cdef class OutputSamplingWithoutReplacementConfig:
    """
    Allows to configure a method for sampling outputs without replacement.
    """

    def get_sample_size(self) -> float:
        """
        Returns the fraction of outputs that are included in a sample.

        :return: The fraction of outputs that are included in a sample
        """
        return self.config_ptr.getSampleSize()

    def set_sample_size(self, sample_size: float) -> OutputSamplingWithoutReplacementConfig:
        """
        Sets the fraction of outputs that should be included in a sample.

        :param sample_size: The fraction of outputs that should be included in a sample, e.g., a value of 0.6
                            corresponds to 60 % of the available outputs. Must be in (0, 1)
        :return:            An `OutputSamplingWithoutReplacementConfig` that allows further configuration of the method
                            for sampling outputs
        """
        assert_greater('sample_size', sample_size, 0)
        assert_less('sample_size', sample_size, 1)
        self.config_ptr.setSampleSize(sample_size)
        return self

    def get_min_samples(self) -> int:
        """
        Returns the minimum number of outputs that are included in a sample.

        :return: The minimum number of outputs that are included in a sample
        """
        return self.config_ptr.getMinSamples()

    def set_min_samples(self, min_samples: int) -> OutputSamplingWithoutReplacementConfig:
        """
        Sets the minimum number of outputs that should be included in a sample.

        :param min_samples: The minimum number of outputs that should be included in a sample. Must be at least 1
        :return:            An `OutputSamplingWithoutReplacementConfig` that allows further configuration of the method
                            for sampling outputs
        """
        assert_greater_or_equal('min_samples', min_samples, 1)
        self.config_ptr.setMinSamples(min_samples)
        return self

    def get_max_samples(self) -> int:
        """
        Returns the maximum number of outputs that are included in a sample.

        :return: The maximum number of outputs that are included in a sample
        """
        return self.config_ptr.getMaxSamples()

    def set_max_samples(self, max_samples: int) -> OutputSamplingWithoutReplacementConfig:
        """
        Sets the maximum number of outputs that should be included in a sample.

        :param max_samples: The maximum number of outputs that should be included in a sample. Must be at least
                            `get_min_samples()` or 0, if the number of outputs should not be restricted
        :return:            An `OutputSamplingWithoutReplacementConfig` that allows further configuration of the method
                            for sampling outputs
        """
        if max_samples != 0:
            assert_greater_or_equal('max_samples', max_samples, self.get_min_samples())
        self.config_ptr.setMaxSamples(max_samples)
        return self

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
