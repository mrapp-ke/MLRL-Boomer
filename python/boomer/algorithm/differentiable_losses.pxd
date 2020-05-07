from boomer.algorithm._arrays cimport uint8, uint32, intp, float64
from boomer.algorithm.losses cimport Loss, Prediction, LabelIndependentPrediction


cdef class DifferentiableLoss(Loss):

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


cdef class DecomposableDifferentiableLoss(DifferentiableLoss):

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


cdef class NonDecomposableDifferentiableLoss(DifferentiableLoss):

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
