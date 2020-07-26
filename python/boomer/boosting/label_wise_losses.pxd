from boomer.common._arrays cimport intp, float64
from boomer.common.statistics cimport LabelMatrix

from libcpp.pair cimport pair


cdef class LabelWiseLoss:

    # Functions:

    cdef pair[float64, float64] calculate_gradient_and_hessian(self, LabelMatrix label_matrix, intp example_index,
                                                               intp label_index, float64 predicted_score)


cdef class LabelWiseLogisticLoss(LabelWiseLoss):

    # Functions:

    cdef pair[float64, float64] calculate_gradient_and_hessian(self, LabelMatrix label_matrix, intp example_index,
                                                               intp label_index, float64 predicted_score)


cdef class LabelWiseSquaredErrorLoss(LabelWiseLoss):

    # Functions:

    cdef pair[float64, float64] calculate_gradient_and_hessian(self, LabelMatrix label_matrix, intp example_index,
                                                               intp label_index, float64 predicted_score)
