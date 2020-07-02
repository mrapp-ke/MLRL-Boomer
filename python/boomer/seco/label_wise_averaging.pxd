from boomer.common._arrays cimport uint8, uint32, intp, float64
from boomer.common.losses cimport RefinementSearch, DecomposableRefinementSearch
from boomer.common.losses cimport DefaultPrediction, Prediction, LabelWisePrediction
from boomer.seco.coverage_losses cimport CoverageLoss
from boomer.seco.heuristics cimport Heuristic


cdef class LabelWiseRefinementSearch(DecomposableRefinementSearch):

    # Attributes:

    cdef Heuristic heuristic

    cdef const uint8[::1, :] true_labels

    cdef const float64[::1, :] uncovered_labels

    cdef const uint8[::1] minority_labels

    cdef const float64[::1, :] confusion_matrices_default

    cdef float64[::1, :] confusion_matrices_covered

    cdef float64[::1, :] accumulated_confusion_matrices_covered

    cdef intp[::1] label_indices

    cdef LabelWisePrediction prediction

    # Functions:

    cdef void update_search(self, intp example_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelWisePrediction calculate_label_wise_prediction(self, bint uncovered, bint accumulated)

    cdef Prediction calculate_example_wise_prediction(self, bint uncovered, bint accumulated)


cdef class LabelWiseAveraging(CoverageLoss):

    # Attributes:

    cdef Heuristic heuristic

    cdef float64[::1, :] uncovered_labels

    cdef uint8[::1] minority_labels

    cdef uint8[::1, :] true_labels

    cdef float64[::1, :] confusion_matrices_default

    # Functions:

    cdef DefaultPrediction calculate_default_prediction(self, uint8[::1, :] y)

    cdef void begin_instance_sub_sampling(self)

    cdef void update_sub_sample(self, intp example_index, uint32 weight, bint remove)

    cdef RefinementSearch begin_search(self, intp[::1] label_indices)

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores)
