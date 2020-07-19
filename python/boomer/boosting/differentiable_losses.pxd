from boomer.common._arrays cimport uint32, intp, float64
from boomer.common.losses cimport LabelMatrix
from boomer.common.losses cimport Loss, RefinementSearch
from boomer.common.losses cimport DefaultPrediction, Prediction, LabelWisePrediction

from libc.math cimport pow


cdef class DifferentiableLoss(Loss):

    # Functions:

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix)

    cdef void begin_instance_sub_sampling(self)

    cdef void update_sub_sample(self, intp example_index, uint32 weight, bint remove)

    cdef RefinementSearch begin_search(self, intp[::1] label_indices)

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores)


cdef inline float64 _l2_norm_pow(float64* a, intp n):
    """
    Computes and returns the square of the L2 norm of a specific vector, i.e. the sum of the squares of its elements. To
    obtain the actual L2 norm, the square-root of the result provided by this function must be computed.

    :param a:   A pointer to an array of type `float64`, shape `(n)`, representing the elements in the vector
    :param n:   The number of elements in the array `a`
    :return:    A scalar of dtype `float64`, representing the square of the L2 of the given vector
    """
    cdef float64 result = 0
    cdef float64 tmp
    cdef intp i

    for i in range(n):
        tmp = a[i]
        tmp = pow(tmp, 2)
        result += tmp

    return result
