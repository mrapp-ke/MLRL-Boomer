from boomer.common._arrays cimport uint8, uint32, intp, float64
from boomer.common.losses cimport DefaultPrediction, Prediction, LabelWisePrediction
from boomer.seco.coverage_losses cimport DecomposableCoverageLoss
from boomer.seco.heuristics cimport Heuristic


cdef class LabelWiseAveraging(DecomposableCoverageLoss):

    # Attributes:

    cdef Prediction prediction

    cdef Heuristic heuristic

    cdef float64[::1, :] uncovered_labels

    cdef uint8[::1] minority_labels

    cdef uint8[::1, :] true_labels

    cdef float64[::1, :] confusion_matrices_default

    cdef float64[::1, :] confusion_matrices_covered

    cdef float64[::1, :] accumulated_confusion_matrices_covered

    cdef intp[::1] label_indices

    # Functions:

    cdef DefaultPrediction calculate_default_prediction(self, uint8[::1, :] y)

    cdef void begin_instance_sub_sampling(self)

    cdef void update_sub_sample(self, intp example_index, uint32 weight, bint remove)

    cdef void begin_search(self, intp[::1] label_indices)

    cdef void update_search(self, intp example_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelWisePrediction calculate_label_wise_prediction(self, bint uncovered, bint accumulated)

    cdef Prediction calculate_example_wise_prediction(self, bint uncovered, bint accumulated)

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores)
