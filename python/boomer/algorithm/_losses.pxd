from boomer.algorithm._model cimport uint8, intp, float64


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

    cdef begin_search(self, intp[::1] label_indices)

    cdef update_search(self, intp r, uint8 weight)

    cdef float64[::1] calculate_scores(self)

    cdef float64[::1] calculate_quality_scores(self)


cdef class DecomposableLoss(Loss):

    # Functions:

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y)

    cdef begin_search(self, intp[::1] label_indices)

    cdef update_search(self, intp r, uint8 weight)

    cdef float64[::1] calculate_scores(self)

    cdef float64[::1] calculate_quality_scores(self)


cdef class SquaredErrorLoss(DecomposableLoss):

    # Attributes:

    cdef float64[::1, :] gradients

    cdef float64[::1] sums_of_gradients

    cdef float64 sum_of_hessians

    cdef intp[::1] label_indices

    cdef float64[::1] scores

    # Functions:

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y)

    cdef begin_search(self, intp[::1] label_indices)

    cdef update_search(self, intp r, uint8 weight)

    cdef float64[::1] calculate_scores(self)

    cdef float64[::1] calculate_quality_scores(self)
