"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._validation import assert_greater_or_equal


cdef class ExampleWiseClassificationPredictorConfig:
    """
    Allows to configure a predictor that predicts known label vectors for given query examples by summing up the scores
    that are provided by an existing rule-based model and comparing the aggregated score vector to the known label
    vectors according to a certain distance measure. The label vector that is closest to the aggregated score vector is
    finally predicted.
    """

    def get_num_threads(self) -> int:
        """
        Returns the number of CPU threads that are used to make predictions for different query examples in parallel.

        :return: The number of CPU threads that are used to make predictions for different query examples in parallel or
                 0, if all available CPU cores are utilized
        """
        return self.config_ptr.getNumThreads()

    def set_num_threads(self, num_threads: int) -> ExampleWiseClassificationPredictorConfig:
        """
        Sets the number of CPU threads that should be used to make predictions for different query examples in parallel.

        :param num_threads: The number of CPU threads that should be used. Must be at least 1 or 0, if all available CPU
                            cores should be utilized
        :return:            An `ExampleWiseClassificationPredictorConfig` that allows further configuration of the
                            predictor
        """
        if num_threads != 0:
            assert_greater_or_equal('num_threads', num_threads, 1)
        self.config_ptr.setNumThreads(num_threads)
        return self


cdef class LabelWiseClassificationPredictorConfig:
    """
    Allows to configure a predictor that predicts whether individual labels of given query examples are relevant or
    irrelevant by summing up the scores that are provided by the individual rules of an existing rule-based model and
    transforming them into binary values according to a certain threshold that is applied to each label individually (1
    if a score exceeds the threshold, i.e., the label is relevant, 0 otherwise).
    """

    def get_num_threads(self) -> int:
        """
        Returns the number of CPU threads that are used to make predictions for different query examples in parallel.

        :return: The number of CPU threads that are used to make predictions for different query examples in parallel or
                 0, if all available CPU cores are utilized
        """
        return self.config_ptr.getNumThreads()

    def set_num_threads(self, num_threads: int) -> LabelWiseClassificationPredictorConfig:
        """
        Sets the number of CPU threads that should be used to make predictions for different query examples in parallel.

        :param num_threads: The number of CPU threads that should be used. Must be at least 1 or 0, if all available CPU
                            cores should be utilized
        :return:            A `LabelWiseClassificationPredictorConfig` that allows further configuration of the
                            predictor
        """
        if num_threads != 0:
            assert_greater_or_equal('num_threads', num_threads, 1)
        self.config_ptr.setNumThreads(num_threads)
        return self


cdef class LabelWiseRegressionPredictorConfig:
    """
    Allows to configure predictors that predict label-wise regression scores for given query examples by summing up the
    scores that are provided by the individual rules of an existing rule-based model for each label individually.
    """

    def get_num_threads(self) -> int:
        """
        Returns the number of CPU threads that are used to make predictions for different query examples in parallel.

        :return: The number of CPU threads that are used to make predictions for different query examples in parallel or
                 0, if all available CPU cores are utilized
        """
        return self.config_ptr.getNumThreads()

    def set_num_threads(self, num_threads: int) -> LabelWiseRegressionPredictorConfig:
        """
        Sets the number of CPU threads that should be used to make predictions for different query examples in parallel.

        :param num_threads: The number of CPU threads that should be used. Must be at least 1 or 0, if all available CPU
                            cores should be utilized
        :return:            A `LabelWiseRegressionPredictorConfig` that allows further configuration of the predictor
        """
        if num_threads != 0:
            assert_greater_or_equal('num_threads', num_threads, 1)
        self.config_ptr.setNumThreads(num_threads)
        return self


cdef class LabelWiseProbabilityPredictorConfig:
    """
    Allows to configure a predictor that predicts label-wise probabilities for given query examples, which estimate the
    chance of individual labels to be relevant, by summing up the scores that are provided by individual rules of an
    existing rule-based models and transforming the aggregated scores into probabilities in [0, 1] according to a
    certain transformation function that is applied to each label individually.
    """

    def get_num_threads(self) -> int:
        """
        Returns the number of CPU threads that are used to make predictions for different query examples in parallel.

        :return: The number of CPU threads that are used to make predictions for different query examples in parallel or
                 0, if all available CPU cores are utilized
        """
        return self.config_ptr.getNumThreads()

    def set_num_threads(self, num_threads: int) -> LabelWiseProbabilityPredictorConfig:
        """
        Sets the number of CPU threads that should be used to make predictions for different query examples in parallel.

        :param num_threads: The number of CPU threads that should be used. Must be at least 1 or 0, if all available CPU
                            cores should be utilized
        :return:            A `LabelWiseProbabilityPredictorConfig` that allows further configuration of the predictor
        """
        if num_threads != 0:
            assert_greater_or_equal('num_threads', num_threads, 1)
        self.config_ptr.setNumThreads(num_threads)
        return self
