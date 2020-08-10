from boomer.seco.statistics cimport CoverageStatistics, AbstractCoverageStatistics


cdef class UncoveredLabelsCriterion(StoppingCriterion):
    """
    A stopping criterion that stops when the sum of the weight matrix stored by `CoverageStatistics` is smaller than or
    equal to a certain threshold.
    """

    def __cinit__(self, CoverageStatistics statistics, float64 threshold):
        """
        :param loss:        The `CoverageStatistics`
        :param threshold:   The threshold
        """
        self.threshold = threshold
        self.statistics_ptr = statistics.statistics_ptr

    cdef bint should_continue(self, intp num_rules):
        cdef AbstractCoverageStatistics* statistics = <AbstractCoverageStatistics*>self.statistics_ptr.get()
        cdef float64 sum_uncovered_labels = statistics.sumUncoveredLabels_
        cdef float64 threshold = self.threshold
        return sum_uncovered_labels > threshold
