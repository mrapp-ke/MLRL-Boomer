from boomer.common._arrays cimport intp, uint8, uint32, float64
from boomer.common.statistics cimport LabelMatrix, RefinementSearch, DecomposableRefinementSearch
from boomer.common.head_refinement cimport HeadCandidate
from boomer.common.rule_evaluation cimport DefaultPrediction, Prediction, LabelWisePrediction
from boomer.seco.coverage_statistics cimport CoverageStatistics


cdef class LabelWiseRefinementSearch(DecomposableRefinementSearch):

    # Attributes:

    cdef const intp[::1] label_indices

    cdef LabelMatrix label_matrix

    cdef const float64[::1, :] uncovered_labels

    cdef const uint8[::1] minority_labels

    cdef const float64[::1, :] confusion_matrices_default

    cdef const float64[::1, :] confusion_matrices_subsample_default

    cdef float64[::1, :] confusion_matrices_covered

    cdef float64[::1, :] accumulated_confusion_matrices_covered

    # Functions:

    cdef void update_search(self, intp statistic_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated)

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated)


cdef class LabelWiseStatistics(CoverageStatistics):

    # Attributes:

    cdef LabelMatrix label_matrix

    cdef float64[::1, :] uncovered_labels

    cdef uint8[::1] minority_labels

    cdef float64[::1, :] confusion_matrices_default

    cdef float64[::1, :] confusion_matrices_subsample_default

    # Functions:

    cdef void apply_default_prediction(self, LabelMatrix label_matrix, DefaultPrediction* default_prediction)

    cdef void reset_statistics(self)

    cdef void add_sampled_statistic(self, intp statistic_index, uint32 weight)

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove)

    cdef RefinementSearch begin_search(self, intp[::1] label_indices)

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head)
