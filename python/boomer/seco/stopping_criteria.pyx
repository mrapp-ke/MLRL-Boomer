from boomer.seco.statistics cimport AbstractCoverageStatistics


cdef class UncoveredLabelsCriterion(StoppingCriterion):
    """
    A stopping criterion that stops when the sum of the weight matrix stored by `AbstractCoverageStatistics` is smaller
    than or equal to a certain threshold.
    """

    def __cinit__(self, float64 threshold):
        """
        :param threshold: The threshold
        """
        self.threshold = threshold

    cdef bint should_continue(self, AbstractStatistics* statistics, intp num_rules):
        cdef AbstractCoverageStatistics* coverage_statistics = <AbstractCoverageStatistics*>statistics
        cdef float64 sum_uncovered_labels = coverage_statistics.sumUncoveredLabels_
        cdef float64 threshold = self.threshold
        return sum_uncovered_labels > threshold
