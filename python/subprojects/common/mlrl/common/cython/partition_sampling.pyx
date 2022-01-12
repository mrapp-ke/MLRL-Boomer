from mlrl.common.cython._validation import assert_greater, assert_less


cdef class ExampleWiseStratifiedBiPartitionSamplingConfig:
    """
    A wrapper for the C++ class `ExampleWiseStratifiedBiPartitionSamplingConfig`.
    """

    def set_holdout_set_size(self, holdout_set_size: float) -> ExampleWiseStratifiedBiPartitionSamplingConfig:
        """
        Sets the fraction of examples that should be included in the holdout set.

        :param holdout_set_size:    The fraction of examples that should be included in the holdout set, e.g., a value
                                    of 0.6 corresponds to 60 % of the available examples. Must be in (0, 1)
        :return:                    An `ExampleWiseStratifiedBiPartitionSamplingConfig` that allows further
                                    configuration of the method for partitioning the available training examples into
                                    a training set and a holdout set
        """
        assert_greater('holdout_set_size', holdout_set_size, 0)
        assert_less('holdout_set_size', holdout_set_size, 1)
        self.config_ptr.setHoldoutSetSize(holdout_set_size)
        return self
