from boomer.algorithm._model cimport uint8, uint32, intp, float64


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

cdef inline intp __get_label_index(intp i, intp[::1] label_indices):
    """
    Retrieves and returns the index of the i-th label from an array of label indices, if such an array is available.
    Otherwise i is returned.

    :param i:               The position of the label whose index should be retrieved
    :param label_indices:   An array of the dtype int, shape `(num_labels)`, representing the indices of labels
    :return:                A scalar of dtype int, representing the index of the i-th label
    """
    if label_indices is None:
        return i
    else:
        return label_indices[i]


cdef class Loss:

    # Functions:

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y)

    cdef begin_instance_sub_sampling(self)

    cdef update_sub_sample(self, intp r)

    cdef begin_search(self, intp[::1] label_indices)

    cdef update_search(self, intp r, uint32 weight)

    cdef float64[::1] calculate_scores(self, bint covered)

    cdef float64[::1] calculate_quality_scores(self, bint covered)


cdef class DecomposableLoss(Loss):

    # Functions:

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y)

    cdef begin_instance_sub_sampling(self)

    cdef update_sub_sample(self, intp r)

    cdef begin_search(self, intp[::1] label_indices)

    cdef update_search(self, intp r, uint32 weight)

    cdef float64[::1] calculate_scores(self, bint covered)

    cdef float64[::1] calculate_quality_scores(self, bint covered)


cdef class SquaredErrorLoss(DecomposableLoss):

    # Attributes:

    cdef float64[::1, :] gradients

    cdef float64[::1] total_sums_of_gradients

    cdef float64 total_sum_of_hessians

    cdef float64[::1] sums_of_gradients

    cdef float64 sum_of_hessians

    cdef intp[::1] label_indices

    cdef float64[::1] scores

    cdef float64[::1] quality_scores

    # Functions:

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y)

    cdef begin_instance_sub_sampling(self)

    cdef update_sub_sample(self, intp r)

    cdef begin_search(self, intp[::1] label_indices)

    cdef update_search(self, intp r, uint32 weight)

    cdef float64[::1] calculate_scores(self, bint covered)

    cdef float64[::1] calculate_quality_scores(self, bint covered)
