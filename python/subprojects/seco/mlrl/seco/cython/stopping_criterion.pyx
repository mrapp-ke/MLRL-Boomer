"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._validation import assert_greater_or_equal


cdef class CoverageStoppingCriterionConfig:
    """
    Allows to configure a stopping criterion that stops the induction of rules as soon as the sum of the weights of the
    uncovered labels is smaller or equal to a certain threshold.
    """

    def set_threshold(self, threshold: float) -> CoverageStoppingCriterionConfig:
        """
        Sets the threshold that should be used by the stopping criterion.

        :param max_rules:   The threshold that should be used by the stopping criterion. Must be at least 0
        :return:            A `CoverageStoppingCriterionConfig` that allows further configuration of the stopping
                            criterion
        """
        assert_greater_or_equal('threshold', threshold, 0)
        self.config_ptr.setThreshold(threshold)
        return self
