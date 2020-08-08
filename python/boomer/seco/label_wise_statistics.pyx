"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to store the elements of confusion matrices that are computed independently for each label.
"""


cdef class LabelWiseStatistics(CoverageStatistics):
    """
    A wrapper for the C++ class `LabelWiseStatisticsImpl`.
    """

    def __cinit__(self, LabelWiseRuleEvaluation rule_evaluation):
        """
        :param rule_evaluation: The `LabelWiseRuleEvaluation` to be used for calculating the predictions, as well as
                                corresponding quality scores, of rules
        """
        self.statistics = new LabelWiseStatisticsImpl(rule_evaluation.rule_evaluation)

    def __dealloc__(self):
        del self.statistics

    cdef float64 get_sum_uncovered_labels(self):
        cdef AbstractCoverageStatistics* statistics = <AbstractCoverageStatistics*>self.statistics
        return statistics.sumUncoveredLabels_

    cdef void apply_default_prediction(self, LabelMatrix label_matrix, DefaultPrediction* default_prediction):
        cdef AbstractStatistics* statistics = self.statistics
        statistics.applyDefaultPrediction(label_matrix.label_matrix, default_prediction)

    cdef void reset_sampled_statistics(self):
        cdef AbstractStatistics* statistics = self.statistics
        statistics.resetSampledStatistics()

    cdef void add_sampled_statistic(self, intp statistic_index, uint32 weight):
        cdef AbstractStatistics* statistics = self.statistics
        statistics.addSampledStatistic(statistic_index, weight)

    cdef void reset_covered_statistics(self):
        cdef AbstractStatistics* statistics = self.statistics
        statistics.resetCoveredStatistics()

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove):
        cdef AbstractStatistics* statistics = self.statistics
        statistics.updateCoveredStatistic(statistic_index, weight, remove)

    cdef AbstractRefinementSearch* begin_search(self, intp[::1] label_indices):
        cdef intp num_predictions = 0 if label_indices is None else label_indices.shape[0]
        cdef const intp* label_indices_ptr = <const intp*>NULL if label_indices is None else &label_indices[0]
        cdef AbstractStatistics* statistics = self.statistics
        return statistics.beginSearch(num_predictions, label_indices_ptr)

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head):
        cdef const intp* label_indices_ptr = <const intp*>NULL if label_indices is None else &label_indices[0]
        cdef AbstractStatistics* statistics = self.statistics
        statistics.applyPrediction(statistic_index, label_indices_ptr, head)
