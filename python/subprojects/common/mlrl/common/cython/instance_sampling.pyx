"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.validation import assert_greater, assert_greater_or_equal, assert_less


cdef class ExampleWiseStratifiedInstanceSamplingConfig:
    """
    Allows to configure a method for selecting a subset of the available training examples using stratification, where
    distinct label vectors are treated as individual classes.
    """

    def get_sample_size(self) -> float:
        """
        Returns the fraction of examples that are included in a sample.

        :return: The fraction of examples that are included in a sample
        """
        return self.config_ptr.getSampleSize()

    def set_sample_size(self, sample_size: float) -> ExampleWiseStratifiedInstanceSamplingConfig:
        """
        Sets the fraction of examples that should be included in a sample.

        :param sample_size: The fraction of examples that should be included in a sample, e.g., a value of 0.6
                            corresponds to 60 % of the available training examples. Must be in (0, 1)
        :return:            An `ExampleWiseStratifiedInstanceSamplingConfig` that allows further configuration of the
                            method for sampling instances
        """
        assert_greater('sample_size', sample_size, 0)
        assert_less('sample_size', sample_size, 1)
        self.config_ptr.setSampleSize(sample_size)
        return self


cdef class OutputWiseStratifiedInstanceSamplingConfig:
    """
    Allows to configure a method for selecting a subset of the available training examples using stratification, such
    that for each label the proportion of relevant and irrelevant examples is maintained.
    """

    def get_sample_size(self) -> float:
        """
        Returns the fraction of examples that are included in a sample.

        :return: The fraction of examples that are included in a sample
        """
        return self.config_ptr.getSampleSize()

    def set_sample_size(self, sample_size: float) -> OutputWiseStratifiedInstanceSamplingConfig:
        """
        Sets the fraction of examples that should be included in a sample.

        :param sample_size: The fraction of examples that should be included in a sample, e.g., a value of 0.6
                            corresponds to 60 % of the available training examples. Must be in (0, 1)
        :return:            An `OutputWiseStratifiedInstanceSamplingConfig` that allows further configuration of the
                            method for sampling instances
        """
        assert_greater('sample_size', sample_size, 0)
        assert_less('sample_size', sample_size, 1)
        self.config_ptr.setSampleSize(sample_size)
        return self


cdef class InstanceSamplingWithReplacementConfig:
    """
    Allows to configure a method for selecting a subset of the available training examples with replacement.
    """

    def get_sample_size(self) -> float:
        """
        Returns the fraction of examples that are included in a sample.

        :return: The fraction of examples that are included in a sample
        """
        return self.config_ptr.getSampleSize()

    def set_sample_size(self, sample_size: float) -> InstanceSamplingWithReplacementConfig:
        """
        Sets the fraction of examples that should be included in a sample.

        :param sample_size: The fraction of examples that should be included in a sample, e.g., a value of 0.6
                            corresponds to 60 % of the available training examples. Must be in (0, 1)
        :return:            An `InstanceSamplingWithReplacementConfig` that allows further configuration of the method
                            for sampling instances
        """
        assert_greater('sample_size', sample_size, 0)
        assert_less('sample_size', sample_size, 1)
        self.config_ptr.setSampleSize(sample_size)
        return self

    def get_min_samples(self) -> int:
        """
        Returns the minimum number of examples that are included in a sample.

        :return: The minimum number of examples that are included in a sample
        """
        return self.config_ptr.getMinSamples()

    def set_min_samples(self, min_samples: int) -> InstanceSamplingWithReplacementConfig:
        """
        Sets the minimum number of examples that should be included in a sample.

        :param min_samples: The minimum number of examples that should be included in a sample. Must be at least 1
        :return:            An `InstanceSamplingWithReplacementConfig` that allows further configuration of the method
                            for sampling instances
        """
        assert_greater_or_equal('min_samples', min_samples, 1)
        self.config_ptr.setMinSamples(min_samples)
        return self

    def get_max_samples(self) -> int:
        """
        Returns the maximum number of examples that are included in a sample.

        :return: The maximum number of examples that are included in a sample
        """
        return self.config_ptr.getMaxSamples()

    def set_max_samples(self, max_samples: int) -> InstanceSamplingWithReplacementConfig:
        """
        Sets the maximum number of examples that should be included in a sample.

        :param max_samples: The maximum number of examples that should be included in a sample. Must be at least
                            `get_min_samples()` or 0, if the number of examples should not be restricted
        :return:            An `InstanceSamplingWithReplacementConfig` that allows further configuration of the method
                            for sampling instances
        """
        if max_samples != 0:
            assert_greater_or_equal('max_samples', max_samples, self.get_min_samples())
        self.config_ptr.setMaxSamples(max_samples)
        return self


cdef class InstanceSamplingWithoutReplacementConfig:
    """
    Allows to configure a method for selecting a subset of the available training examples without replacement.
    """

    def get_sample_size(self) -> float:
        """
        Returns the fraction of examples that are included in a sample.

        :return: The fraction of examples that are included in a sample
        """
        return self.config_ptr.getSampleSize()

    def set_sample_size(self, sample_size: float) -> InstanceSamplingWithoutReplacementConfig:
        """
        Sets the fraction of examples that should be included in a sample.

        :param sample_size: The fraction of examples that should be included in a sample, e.g., a value of 0.6
                            corresponds to 60 % of the available training examples. Must be in (0, 1)
        :return:            An `InstanceSamplingWithoutReplacementConfig` that allows further configuration of the
                            method for sampling instances
        """
        assert_greater('sample_size', sample_size, 0)
        assert_less('sample_size', sample_size, 1)
        self.config_ptr.setSampleSize(sample_size)
        return self
