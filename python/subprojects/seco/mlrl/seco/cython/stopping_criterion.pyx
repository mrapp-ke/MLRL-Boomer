"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._validation import assert_greater_or_equal


cdef class CoverageStoppingCriterionConfig:
    """
    A wrapper for the C++ class `CoverageStoppingCriterionConfig`.
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
