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


cdef class ExampleWiseLossFunction:
    """
    A base class for all (non-decomposable) loss functions that are applied example-wise.
    """

    cdef void calculate_gradients_and_hessians(self, LabelMatrix label_matrix, intp example_index,
                                               float64* predicted_scores, float64[::1] gradients,
                                               float64[::1] hessians):
        """
        Must be implemented by subclasses to calculate the gradients (first derivatives) and Hessians (second
        derivatives) of the loss function for each label of a certain example.

        :param label_matrix:        A `LabelMatrix` that provides random access to the labels of the training examples
        :param example_index:       The index of the example for which the gradients and Hessians should be calculated
        :param predicted_scores:    A pointer to an array of type `float64`, shape `(num_labels)`, representing the
                                    scores that are predicted for each label of the respective example
        :param gradients:           An array of dtype `float64`, shape `(num_labels)`, the gradients that have been
                                    calculated should be written to
        :param hessians:            An array of dtype `float64`, shape `(num_labels * (num_labels + 1) / 2)`, the
                                    Hessians that have been calculated should be written to
        """
        pass


cdef class ExampleWiseLogisticLossFunction(ExampleWiseLossFunction):
    """
    A multi-label variant of the logistic loss that is applied example-wise.
    """

    # Functions:

    cdef void calculate_gradients_and_hessians(self, LabelMatrix label_matrix, intp example_index,
                                               float64* predicted_scores, float64[::1] gradients,
                                               float64[::1] hessians):
        cdef intp num_labels = label_matrix.num_labels
        cdef float64 sum_of_exponentials = 1
        # Temporary variables
        cdef float64 expected_score, expected_score2, predicted_score, predicted_score2, exponential, tmp
        cdef uint8 true_label
        cdef intp c, c2

        for c in range(num_labels):
            true_label = label_matrix.get_label(example_index, c)
            expected_score = 1 if true_label else -1
            predicted_score = predicted_scores[c]
            exponential = exp(-expected_score * predicted_score)
            gradients[c] = exponential  # Temporarily store the exponential in the existing output array
            sum_of_exponentials += exponential

        cdef float64 sum_of_exponentials_pow = pow(sum_of_exponentials, 2)
        cdef intp j = 0

        for c in range(num_labels):
            true_label = label_matrix.get_label(example_index, c)
            expected_score = 1 if true_label else -1
            predicted_score = predicted_scores[c]
            exponential = gradients[c]

            tmp = (-expected_score * exponential) / sum_of_exponentials
            gradients[c] = tmp

            for c2 in range(c):
                true_label = label_matrix.get_label(example_index, c2)
                expected_score2 = 1 if true_label else -1
                predicted_score2 = predicted_scores[c2]
                tmp = exp((-expected_score2 * predicted_score2) - (expected_score * predicted_score))
                tmp = -expected_score2 * expected_score * tmp
                tmp = tmp / sum_of_exponentials_pow
                hessians[j] = tmp
                j += 1

            tmp = (pow(expected_score, 2) * exponential * (sum_of_exponentials - exponential))
            tmp = tmp / sum_of_exponentials_pow
            hessians[j] = tmp
            j += 1
