from boomer.algorithm._arrays cimport uint8, uint32, intp, float64
from boomer.algorithm._losses import Loss, Prediction

cdef class LabelWiseMeasure(Loss):

    # Attributes:

    cdef Prediction prediction

    cdef readonly float64[::1, :] uncovered_labels

    cdef uint32[::1] minority_labels

    cdef uint8[::1, :] true_labels

    cdef float64[::1, :] confusion_matrices_default

    cdef float64[::1, :] confusion_matrices_covered

    cdef intp[::1] label_indices

    # Functions:

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y)

    cdef begin_instance_sub_sampling(self)

    cdef update_sub_sample(self, intp example_index)

    cdef begin_search(self, intp[::1] label_indices)

    cdef update_search(self, intp example_index, uint32 weight)

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered)

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered)

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores)

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp)
