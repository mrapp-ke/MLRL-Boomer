"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._validation import assert_greater_or_equal


cdef class ExampleWiseClassificationPredictorConfig:
    """
    A wrapper for the C++ class `ExampleWiseClassificationPredictorConfig`.
    """

    def set_num_threads(self, num_threads: int) -> ExampleWiseClassificationPredictorConfig:
        """
        Sets the number of CPU threads that should be used to make predictions for different query examples in parallel.

        :param num_threads: The number of CPU threads that should be used. Must be at least 1 or 0, if the number of CPU
                            threads should be chosen automatically
        :return:            An `ExampleWiseClassificationPredictorConfig` that allows further configuration of the
                            predictor
        """
        if num_threads != 0:
            assert_greater_or_equal('num_threads', num_threads, 1)
        self.config_ptr.setNumThreads(num_threads)
        return self


cdef class LabelWiseClassificationPredictorConfig:
    """
    A wrapper for the C++ class `LabelWiseClassificationPredictorConfig`.
    """

    def set_num_threads(self, num_threads: int) -> LabelWiseClassificationPredictorConfig:
        """
        Sets the number of CPU threads that should be used to make predictions for different query examples in parallel.

        :param num_threads: The number of CPU threads that should be used. Must be at least 1 or 0, if the number of CPU
                            threads should be chosen automatically
        :return:            A `LabelWiseClassificationPredictorConfig` that allows further configuration of the
                            predictor
        """
        if num_threads != 0:
            assert_greater_or_equal('num_threads', num_threads, 1)
        self.config_ptr.setNumThreads(num_threads)
        return self


cdef class LabelWiseRegressionPredictorConfig:
    """
    A wrapper for the C++ class `LabelWiseRegressionPredictorConfig`.
    """

    def set_num_threads(self, num_threads: int) -> LabelWiseRegressionPredictorConfig:
        """
        Sets the number of CPU threads that should be used to make predictions for different query examples in parallel.

        :param num_threads: The number of CPU threads that should be used. Must be at least 1 or 0, if the number of CPU
                            threads should be chosen automatically
        :return:            A `LabelWiseRegressionPredictorConfig` that allows further configuration of the predictor
        """
        if num_threads != 0:
            assert_greater_or_equal('num_threads', num_threads, 1)
        self.config_ptr.setNumThreads(num_threads)
        return self


cdef class LabelWiseProbabilityPredictorConfig:
    """
    A wrapper for the C++ class `LabelWiseProbabilityPredictorConfig`.
    """

    def set_num_threads(self, num_threads: int) -> LabelWiseProbabilityPredictorConfig:
        """
        Sets the number of CPU threads that should be used to make predictions for different query examples in parallel.

        :param num_threads: The number of CPU threads that should be used. Must be at least 1 or 0, if the number of CPU
                            threads should be chosen automatically
        :return:            A `LabelWiseProbabilityPredictorConfig` that allows further configuration of the predictor
        """
        if num_threads != 0:
            assert_greater_or_equal('num_threads', num_threads, 1)
        self.config_ptr.setNumThreads(num_threads)
        return self
