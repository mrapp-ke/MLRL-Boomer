"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.util.validation import assert_greater_or_equal, assert_less


cdef class CoverageStoppingCriterionConfig:
    """
    Allows to configure a stopping criterion that stops the induction of rules as soon as a certain fraction of the
    available examples and labels is covered.
    """

    def get_min_coverage(self) -> float:
        """
        Returns the fraction of training examples and labels that must be covered before the induction of rules is
        stopped.

        :return: The fraction that must be covered before the induction of rules is stopped
        """
        return self.config_ptr.getMinCoverage()

    def set_min_coverage(self, min_coverage: float) -> CoverageStoppingCriterionConfig:
        """
        Sets the fraction of training examples and labels that must be covered before the induction of rules is stopped.

        :param min_coverage:    The fraction of training examples and labels that must be covered before the induction
                                of rules is stopped. Must be in [0, 1)
        :return:                A `CoverageStoppingCriterionConfig` that allows further configuration of the stopping
                                criterion
        """
        assert_greater_or_equal('min_coverage', min_coverage, 0)
        assert_less('min_coverage', min_coverage, 1)
        self.config_ptr.setMinCoverage(min_coverage)
        return self
