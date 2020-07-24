from boomer.common._arrays cimport uint32, intp, float64
from boomer.common.statistics cimport LabelMatrix, Statistics, RefinementSearch
from boomer.common.head_refinement cimport HeadCandidate
from boomer.common.rule_evaluation cimport DefaultPrediction


cdef class CoverageStatistics(Statistics):

    # Attributes:

    cdef readonly float64 sum_uncovered_labels

    # Functions:

    cdef void apply_default_prediction(self, LabelMatrix label_matrix, DefaultPrediction* default_prediction)

    cdef void reset_sampled_statistics(self)

    cdef void add_sampled_statistic(self, intp statistic_index, uint32 weight)

    cdef void reset_covered_statistics(self)

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove)

    cdef RefinementSearch begin_search(self, intp[::1] label_indices)

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head)
