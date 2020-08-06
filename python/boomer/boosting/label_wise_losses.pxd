from boomer.common._arrays cimport intp, float64
from boomer.common.input_data cimport LabelMatrix, AbstractLabelMatrix

from libcpp.pair cimport pair


cdef extern from "cpp/label_wise_losses.h" namespace "boosting" nogil:

    cdef cppclass AbstractLabelWiseLoss:

        # Functions:

        pair[float64, float64] calculateGradientAndHessian(AbstractLabelMatrix* labelMatrix, intp exampleIndex,
                                                           intp labelIndex, float64 predictedScore)


    cdef cppclass LabelWiseLogisticLossImpl(AbstractLabelWiseLoss):

        # Functions:

        pair[float64, float64] calculateGradientAndHessian(AbstractLabelMatrix* labelMatrix, intp exampleIndex,
                                                           intp labelIndex, float64 predictedScore)


    cdef cppclass LabelWiseSquaredErrorLossImpl(AbstractLabelWiseLoss):

        # Functions:

        pair[float64, float64] calculateGradientAndHessian(AbstractLabelMatrix* labelMatrix, intp exampleIndex,
                                                           intp labelIndex, float64 predictedScore)


cdef class LabelWiseLoss:

    # Attributes:

    cdef AbstractLabelWiseLoss* loss_function

    # Functions:

    cdef pair[float64, float64] calculate_gradient_and_hessian(self, LabelMatrix label_matrix, intp example_index,
                                                               intp label_index, float64 predicted_score) nogil


cdef class LabelWiseLogisticLoss(LabelWiseLoss):

    # Functions:

    cdef pair[float64, float64] calculate_gradient_and_hessian(self, LabelMatrix label_matrix, intp example_index,
                                                               intp label_index, float64 predicted_score) nogil


cdef class LabelWiseSquaredErrorLoss(LabelWiseLoss):

    # Functions:

    cdef pair[float64, float64] calculate_gradient_and_hessian(self, LabelMatrix label_matrix, intp example_index,
                                                               intp label_index, float64 predicted_score) nogil
