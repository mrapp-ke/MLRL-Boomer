from boomer.common._arrays cimport intp, float64
from boomer.common.statistics cimport LabelMatrix


cdef class ExampleWiseLoss:

    # Functions:

    cdef void calculate_gradients_and_hessians(self, LabelMatrix label_matrix, intp example_index,
                                               float64* predicted_scores, float64[::1] gradients, float64[::1] hessians)


cdef class ExampleWiseLogisticLoss(ExampleWiseLoss):

    # Functions:

    cdef void calculate_gradients_and_hessians(self, LabelMatrix label_matrix, intp example_index,
                                               float64* predicted_scores, float64[::1] gradients, float64[::1] hessians)
