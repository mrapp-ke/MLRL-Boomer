"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement different differentiable loss functions.
"""
from boomer.common._arrays cimport uint8

from libc.math cimport exp, pow


cdef class LabelWiseLossFunction:
    """
    A base class for all (decomposable) loss functions that are applied label-wise.
    """

    cdef pair[float64, float64] calculate_gradient_and_hessian(self, LabelMatrix label_matrix, intp example_index,
                                                               intp label_index, float64 predicted_score):
        """
        Must be implemented by subclasses to calculate the gradient (first derivative) and Hessian (second derivative)
        of the loss function for a certain example and label.

        :param label_matrix:    A `LabelMatrix` that provides random access to the labels of the training examples
        :param example_index:   The index of the example for which the gradient and Hessian should be calculated
        :param label_index:     The index of the label for which the gradient and Hessian should be calculated
        :param predicted_score: A scalar of dtype float64, representing the score that is predicted for the respective
                                example and label
        :return:                A pair that contains two scalars of dtype float64, representing the gradient and the
                                Hessian that have been calculated
        """
        pass


cdef class LabelWiseLogisticLossFunction(LabelWiseLossFunction):
    """
    A multi-label variant of the logistic loss that is applied label-wise.
    """

    cdef pair[float64, float64] calculate_gradient_and_hessian(self, LabelMatrix label_matrix, intp example_index,
                                                               intp label_index, float64 predicted_score):
        cdef uint8 true_label = label_matrix.get_label(example_index, label_index)
        cdef float64 expected_score = 1 if true_label else -1
        cdef float64 exponential = exp(expected_score * predicted_score)
        cdef float64 gradient = -expected_score / (1 + exponential)
        cdef float64 hessian = (pow(expected_score, 2) * exponential) / pow(1 + exponential, 2)
        cdef pair[float64, float64] result  # Stack-allocated pair
        result.first = gradient
        result.second = hessian
        return result


cdef class LabelWiseSquaredErrorLossFunction(LabelWiseLossFunction):
    """
    A multi-label variant of the squared error loss that is applied label-wise.
    """

    cdef pair[float64, float64] calculate_gradient_and_hessian(self, LabelMatrix label_matrix, intp example_index,
                                                               intp label_index, float64 predicted_score):
        cdef uint8 true_label = label_matrix.get_label(example_index, label_index)
        cdef float64 expected_score = 1 if true_label else -1
        cdef float64 gradient = 2 * predicted_score - 2 * expected_score
        cdef float64 hessian = 2
        cdef pair[float64, float64] result  # Stack-allocated pair
        result.first = gradient
        result.second = hessian
        return result
