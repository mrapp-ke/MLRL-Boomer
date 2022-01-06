"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.validation import assert_greater_or_equal


cdef class TopDownRuleInductionConfig:
    """
    A wrapper for the C++ class `TopDownRuleInductionConfig`.
    """

    def set_min_coverage(self, min_coverage: int) -> TopDownRuleInductionConfig:
        """
        Sets the minimum number of training examples that must be covered by a rule.

        :param min_coverage:    The minimum number of training examples that must be covered by a rule. Must be at least
                                1
        :return:                A `TopDownRuleInductionConfig` that allows further configuration of the algorithm for
                                the induction of individual rules
        """
        assert_greater_or_equal('min_coverage', min_coverage, 1)
        self.config_ptr.setMinCoverage(min_coverage)
        return self

    def set_max_conditions(self, max_conditions: int) -> TopDownRuleInductionConfig:
        """
        Sets the maximum number of conditions to be included in a rule's body.

        :param max_conditions:  The maximum number of conditions to be included in a rule's body. Must be at least 1 or
                                0, if the number of conditions should not be restricted
        :return:                A `TopDownRuleInductionConfig` that allows further configuration of the algorithm for
                                the induction of individual rules
        """
        if max_conditions != 0:
            assert_greater_or_equal('max_conditions', max_conditions, 1)
        self.config_ptr.setMaxConditions(max_conditions)
        return self

    def set_max_head_refinements(self, max_head_refinements: int) -> TopDownRuleInductionConfig:
        """
        Sets the maximum number of times, the head of a rule may be refined after a new condition has been added to its
        body.

        :param max_head_refinements:    The maximum number of times, the head of a rule may be refined. Must be at least
                                        1 or 0, if the number of refinements should not be restricted
        :return:                        A `TopDownRuleInductionConfig` that allows further configuration of the
                                        algorithm for the induction of individual rules
        """
        if max_head_refinements != 0:
            assert_greater_or_equal('max_head_refinements', max_head_refinements, 1)
        self.config_ptr.setMaxHeadRefinements(max_head_refinements)
        return self

    def set_recalculate_predictions(self, recalculate_predictions: bool) -> TopDownRuleInductionConfig:
        """
        Sets whether the predictions of rules should be recalculated on all training examples, if some of the examples
        have zero weights, or not.

        :param recalculate_predictions: True, if the predictions of rules should be recalculated on all training
                                        examples, False otherwise
        :return:                        A `TopDownRuleInductionConfig` that allows further configuration of the
                                        algorithm for the induction of individual rules
        """
        self.config_ptr.setRecalculatePredictions(recalculate_predictions)
        return self

    def set_num_threads(self, num_threads: int) -> TopDownRuleInductionConfig:
        """
        Sets the number of CPU threads to be used to search for potential refinements of rules in parallel.

        :param num_threads: The number of CPU threads to be used. Must be at least 1 or 0, if the number of threads
                            should be chosen automatically
        :return:            A `TopDownRuleInductionConfig` that allows further configuration of the algorithm for the
                            induction of individual rules
        """
        if num_threads != 0:
            assert_greater_or_equal('num_threads', num_threads, 1)
        self.config_ptr.setNumThreads(num_threads)
        return self
