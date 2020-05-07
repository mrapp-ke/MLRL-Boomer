from boomer.algorithm._arrays cimport uint8, uint32, intp, float64
from boomer.algorithm.heuristics cimport Heuristic
from boomer.algorithm.losses cimport Prediction, LabelIndependentPrediction
from boomer.algorithm.coverage_losses cimport DecomposableCoverageLoss


cdef class LabelWiseAveraging(DecomposableCoverageLoss):

    # Attributes:

    cdef Prediction prediction

    cdef Heuristic heuristic

    cdef float64[::1, :] uncovered_labels
    
    cdef float64 sum_uncovered_labels

    cdef uint8[::1] minority_labels

    cdef uint8[::1, :] true_labels

    cdef float64[::1, :] confusion_matrices_default

    cdef float64[::1, :] confusion_matrices_covered

    cdef intp[::1] label_indices

    # Functions:

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y)

    cdef void begin_instance_sub_sampling(self)

    cdef void update_sub_sample(self, intp example_index)

    cdef void begin_search(self, intp[::1] label_indices)

    cdef void update_search(self, intp example_index, uint32 weight)

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered)

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered)

    cdef void apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                                float64[::1] predicted_scores)
