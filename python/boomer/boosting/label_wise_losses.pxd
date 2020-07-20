from boomer.common._arrays cimport uint32, intp, float64
from boomer.common.statistics cimport LabelMatrix
from boomer.common.losses cimport RefinementSearch, DecomposableRefinementSearch
from boomer.common.rule_evaluation cimport DefaultPrediction, Prediction, LabelWisePrediction
from boomer.common.head_refinement cimport HeadCandidate
from boomer.boosting.differentiable_losses cimport DifferentiableLoss

from libcpp.pair cimport pair


cdef class LabelWiseLossFunction:

    # Functions:

    cdef pair[float64, float64] calculate_gradient_and_hessian(self, LabelMatrix label_matrix, intp example_index,
                                                               intp label_index, float64 predicted_score)


cdef class LabelWiseLogisticLossFunction(LabelWiseLossFunction):

    # Functions:

    cdef pair[float64, float64] calculate_gradient_and_hessian(self, LabelMatrix label_matrix, intp example_index,
                                                               intp label_index, float64 predicted_score)


cdef class LabelWiseSquaredErrorLossFunction(LabelWiseLossFunction):

    # Functions:

    cdef pair[float64, float64] calculate_gradient_and_hessian(self, LabelMatrix label_matrix, intp example_index,
                                                               intp label_index, float64 predicted_score)


cdef class LabelWiseRefinementSearch(DecomposableRefinementSearch):

    # Attributes:

    cdef float64 l2_regularization_weight

    cdef const intp[::1] label_indices

    cdef const float64[:, ::1] gradients

    cdef const float64[::1] total_sums_of_gradients

    cdef float64[::1] sums_of_gradients

    cdef float64[::1] accumulated_sums_of_gradients

    cdef const float64[:, ::1] hessians

    cdef const float64[::1] total_sums_of_hessians

    cdef float64[::1] sums_of_hessians

    cdef float64[::1] accumulated_sums_of_hessians

    cdef LabelWisePrediction* prediction

    # Functions:

    cdef void update_search(self, intp example_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated)

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated)


cdef class LabelWiseDifferentiableLoss(DifferentiableLoss):

    # Attributes:

    cdef LabelWiseLossFunction loss_function

    cdef float64 l2_regularization_weight

    cdef LabelMatrix label_matrix

    cdef float64[:, ::1] current_scores

    cdef float64[:, ::1] gradients

    cdef float64[::1] total_sums_of_gradients

    cdef float64[:, ::1] hessians

    cdef float64[::1] total_sums_of_hessians

    # Functions:

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix)

    cdef void reset_examples(self)

    cdef void add_sampled_example(self, intp example_index, uint32 weight)

    cdef void update_covered_example(self, intp example_index, uint32 weight, bint remove)

    cdef RefinementSearch begin_search(self, intp[::1] label_indices)

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, HeadCandidate* head)
