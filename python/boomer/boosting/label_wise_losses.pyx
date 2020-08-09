"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement different differentiable loss functions.
"""
from boomer.common._arrays cimport uint8

from libc.math cimport exp, pow

from libcpp.memory cimport make_shared


cdef class LabelWiseLoss:
    """
    A wrapper for the abstract C++ class `AbstractLabelWiseLoss`.
    """

    cdef pair[float64, float64] calculate_gradient_and_hessian(self, LabelMatrix label_matrix, intp example_index,
                                                               intp label_index, float64 predicted_score) nogil:
        """
        Calculates the gradient (first derivative) and Hessian (second derivative) of the loss function for a certain
        example and label.

        :param label_matrix:    A `LabelMatrix` that provides random access to the labels of the training examples
        :param example_index:   The index of the example for which the gradient and Hessian should be calculated
        :param label_index:     The index of the label for which the gradient and Hessian should be calculated
        :param predicted_score: A scalar of dtype float64, representing the score that is predicted for the respective
                                example and label
        :return:                A pair that contains two scalars of dtype float64, representing the gradient and the
                                Hessian that have been calculated
        """
        cdef AbstractLabelWiseLoss* loss_function = self.loss_function_ptr.get()
        cdef AbstractLabelMatrix* label_matrix_ptr = label_matrix.label_matrix
        return loss_function.calculateGradientAndHessian(label_matrix_ptr, example_index, label_index, predicted_score)


cdef class LabelWiseLogisticLoss(LabelWiseLoss):
    """
    A wrapper for the C++ class `LabelWiseLogisticLossImpl`.
    """

    def __cinit__(self):
        self.loss_function_ptr = <shared_ptr[AbstractLabelWiseLoss]>make_shared[LabelWiseLogisticLossImpl]()


cdef class LabelWiseSquaredErrorLoss(LabelWiseLoss):
    """
    A wrapper for the C++ class `LabelWiseSquaredErrorLossImpl`.
    """

    def __cinit__(self):
        self.loss_function_ptr = <shared_ptr[AbstractLabelWiseLoss]>make_shared[LabelWiseSquaredErrorLossImpl]()
