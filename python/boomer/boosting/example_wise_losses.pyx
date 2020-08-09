"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement different differentiable loss functions.
"""
from boomer.common._arrays cimport uint8

from libc.math cimport exp, pow

from libcpp.memory cimport make_shared


cdef class ExampleWiseLoss:
    """
    A wrapper for the abstract C++ class `AbstractExampleWiseLoss`.
    """

    cdef void calculate_gradients_and_hessians(self, LabelMatrix label_matrix, intp example_index,
                                               float64* predicted_scores, float64[::1] gradients,
                                               float64[::1] hessians) nogil:
        """
        Calculates the gradients (first derivatives) and Hessians (second derivatives) of the loss function for each
        label of a certain example.

        :param label_matrix:        A `LabelMatrix` that provides random access to the labels of the training examples
        :param example_index:       The index of the example for which the gradients and Hessians should be calculated
        :param predicted_scores:    A pointer to an array of type `float64`, shape `(num_labels)`, representing the
                                    scores that are predicted for each label of the respective example
        :param gradients:           An array of dtype `float64`, shape `(num_labels)`, the gradients that have been
                                    calculated should be written to
        :param hessians:            An array of dtype `float64`, shape `(num_labels * (num_labels + 1) / 2)`, the
                                    Hessians that have been calculated should be written to
        """
        cdef AbstractExampleWiseLoss* loss_function = self.loss_function_ptr.get()
        cdef AbstractLabelMatrix* label_matrix_ptr = label_matrix.label_matrix
        loss_function.calculateGradientsAndHessians(label_matrix_ptr, example_index, predicted_scores, &gradients[0],
                                                    &hessians[0])


cdef class ExampleWiseLogisticLoss(ExampleWiseLoss):
    """
    A wrapper for the C++ class `ExampleWiseLogisticLossImpl`.
    """

    def __cinit__(self):
        self.loss_function_ptr = <shared_ptr[AbstractExampleWiseLoss]>make_shared[ExampleWiseLogisticLossImpl]()
