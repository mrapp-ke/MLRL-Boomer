from boomer.common._arrays cimport uint8, uint32, intp, float64
from boomer.common.losses cimport RefinementSearch, NonDecomposableRefinementSearch
from boomer.common.losses cimport DefaultPrediction, Prediction, LabelWisePrediction
from boomer.boosting.differentiable_losses cimport DifferentiableLoss


cdef class ExampleWiseLossFunction:

    # Functions:

    cdef void calculate_gradients_and_hessians(self, uint8[::1] true_labels, float64[::1] predicted_scores,
                                               float64[::1] gradients, float64[::1] hessians)


cdef class ExampleWiseLogisticLossFunction(ExampleWiseLossFunction):

    # Functions:

    cdef void calculate_gradients_and_hessians(self, uint8[::1] true_labels, float64[::1] predicted_scores,
                                               float64[::1] gradients, float64[::1] hessians)


cdef class ExampleWiseLogisticLossRefinementSearch(NonDecomposableRefinementSearch):

    # Attributes:

    cdef float64 l2_regularization_weight

    cdef const intp[::1] label_indices

    cdef const float64[::1, :] gradients

    cdef const float64[::1] total_sums_of_gradients

    cdef float64[::1] sums_of_gradients

    cdef float64[::1] accumulated_sums_of_gradients

    cdef const float64[::1, :] hessians

    cdef const float64[::1] total_sums_of_hessians

    cdef float64[::1] sums_of_hessians

    cdef float64[::1] accumulated_sums_of_hessians

    cdef LabelWisePrediction prediction

    # Functions:

    cdef void update_search(self, intp example_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelWisePrediction calculate_label_wise_prediction(self, bint uncovered, bint accumulated)

    cdef Prediction calculate_example_wise_prediction(self, bint uncovered, bint accumulated)


cdef class ExampleWiseLogisticLoss(DifferentiableLoss):

    # Attributes:

    cdef float64 l2_regularization_weight

    cdef uint8[:, ::1] true_labels

    cdef float64[::1, :] current_scores

    cdef float64[::1, :] gradients

    cdef float64[::1] total_sums_of_gradients

    cdef float64[::1, :] hessians

    cdef float64[::1] total_sums_of_hessians

    # Functions:

    cdef DefaultPrediction calculate_default_prediction(self, uint8[:, ::1] y)

    cdef void begin_instance_sub_sampling(self)

    cdef void update_sub_sample(self, intp example_index, uint32 weight, bint remove)

    cdef RefinementSearch begin_search(self, intp[::1] label_indices)

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores)
