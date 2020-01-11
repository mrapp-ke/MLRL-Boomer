from boomer.algorithm._arrays cimport uint8, uint32, intp, float64


# Utility functions:

cdef inline float64 __convert_label_into_score(uint8 label):
    """
    Converts a label in {0, 1} into an expected score {-1, 1}.

    :param label:   A scalar of dtype float, representing the label
    :return:        A scalar of dtype float, representing the confidence score
    """
    if label > 0:
        return label
    else:
        return -1


cdef class Loss:

    # Functions:

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y)

    cdef begin_instance_sub_sampling(self)

    cdef update_sub_sample(self, intp example_index)

    cdef begin_search(self, intp[::1] label_indices)

    cdef update_search(self, intp example_index, uint32 weight)

    cdef float64[::1, :] calculate_predicted_and_quality_scores(self, bint include_uncovered)

    cdef float64[::1] calculate_predicted_scores(self)

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores)


cdef class DecomposableLoss(Loss):

    # Functions:

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y)

    cdef begin_instance_sub_sampling(self)

    cdef update_sub_sample(self, intp example_index)

    cdef begin_search(self, intp[::1] label_indices)

    cdef update_search(self, intp example_index, uint32 weight)

    cdef float64[::1, :] calculate_predicted_and_quality_scores(self, bint include_uncovered)

    cdef float64[::1] calculate_predicted_scores(self)

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores)


cdef class NonDecomposableLoss(Loss):

    # Functions:

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y)

    cdef begin_instance_sub_sampling(self)

    cdef update_sub_sample(self, intp example_index)

    cdef begin_search(self, intp[::1] label_indices)

    cdef update_search(self, intp example_index, uint32 weight)

    cdef float64[::1, :] calculate_predicted_and_quality_scores(self, bint include_uncovered)

    cdef float64[::1] calculate_predicted_scores(self)

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores)


cdef class SquaredErrorLoss(DecomposableLoss):

    # Attributes:

    cdef float64[::1, :] gradients

    cdef float64[::1] sums_of_gradients

    cdef float64[::1] total_sums_of_gradients

    cdef float64 sum_of_hessians

    cdef float64 total_sum_of_hessians

    cdef intp[::1] label_indices

    cdef float64[::1, :] predicted_and_quality_scores

    # Functions:

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y)

    cdef begin_instance_sub_sampling(self)

    cdef update_sub_sample(self, intp example_index)

    cdef begin_search(self, intp[::1] label_indices)

    cdef update_search(self, intp example_index, uint32 weight)

    cdef float64[::1, :] calculate_predicted_and_quality_scores(self, bint include_uncovered)

    cdef float64[::1] calculate_predicted_scores(self)

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores)


cdef class LogisticLoss(NonDecomposableLoss):

    # Attributes:

    cdef float64[::1, :] gradients

    cdef float64[::1] sums_of_gradients

    cdef float64[::1] total_sums_of_gradients

    cdef float64[::1, :] hessians

    cdef float64[::1] sums_of_hessians

    cdef float64[::1] total_sums_of_hessians

    cdef intp[::1] label_indices

    # Functions:

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y)

    cdef begin_instance_sub_sampling(self)

    cdef update_sub_sample(self, intp example_index)

    cdef begin_search(self, intp[::1] label_indices)

    cdef update_search(self, intp example_index, uint32 weight)

    cdef float64[::1, :] calculate_predicted_and_quality_scores(self, bint include_uncovered)

    cdef float64[::1] calculate_predicted_scores(self)

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores)
