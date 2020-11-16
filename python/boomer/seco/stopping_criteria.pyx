"""
@author Jakob Steeg (jakob.steeg@gmail.com)
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement different stopping criteria for separate-and-conquer algorithms.
"""
from boomer.seco.statistics cimport ICoverageStatistics


cdef class UncoveredLabelsCriterion(StoppingCriterion):
    """
    A stopping criterion that stops when the sum of the weight matrix stored by `ICoverageStatistics` is smaller than or
    equal to a certain threshold.
    """

    def __cinit__(self, float64 threshold):
        """
        :param threshold: The threshold
        """
        self.threshold = threshold

    cdef bint should_continue(self, IStatistics* statistics, uint32 num_rules):
        cdef ICoverageStatistics* coverage_statistics = <ICoverageStatistics*>statistics
        cdef float64 sum_uncovered_labels = coverage_statistics.getSumOfUncoveredLabels()
        cdef float64 threshold = self.threshold
        return sum_uncovered_labels > threshold
