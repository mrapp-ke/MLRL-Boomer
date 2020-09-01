"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to store gradients and Hessians that are calculated according to a
(non-decomposable) loss function that is applied example-wise.
"""
from boomer.common.input_data cimport RandomAccessLabelMatrix, AbstractLabelMatrix
from boomer.boosting._lapack cimport init_lapack
from boomer.boosting.example_wise_losses cimport ExampleWiseLoss
from boomer.boosting.example_wise_rule_evaluation cimport ExampleWiseRuleEvaluation

from libcpp.memory cimport unique_ptr, dynamic_pointer_cast


cdef class ExampleWiseStatisticsFactory(StatisticsFactory):
    """
    A wrapper for the C++ class `ExampleWiseStatisticsFactoryImpl`.
    """

    def __cinit__(self, ExampleWiseLoss loss_function, ExampleWiseRuleEvaluation rule_evaluation):
        """
        :param loss_function:   The loss function to be used for calculating gradients and Hessians
        :param rule_evaluation: The `ExampleWiseRuleEvaluation` to be used for calculating the predictions, as well as
                                corresponding quality scores, of rules
        :param label_matrix:    A label matrix that provides random access to the labels of the training examples
        """
        self.loss_function_ptr = loss_function.loss_function_ptr
        self.rule_evaluation_ptr = rule_evaluation.rule_evaluation_ptr
        cdef shared_ptr[Lapack] lapack_ptr
        lapack_ptr.reset(init_lapack())
        self.lapack_ptr = lapack_ptr

    cdef AbstractStatistics* create(self, LabelMatrix label_matrix):
        cdef unique_ptr[AbstractStatisticsFactory] statistics_factory_ptr

        if isinstance(label_matrix, RandomAccessLabelMatrix):
            statistics_factory_ptr.reset(new ExampleWiseStatisticsFactoryImpl(
                self.loss_function_ptr, self.rule_evaluation_ptr, self.lapack_ptr,
                dynamic_pointer_cast[AbstractRandomAccessLabelMatrix, AbstractLabelMatrix](
                    label_matrix.label_matrix_ptr)))
        else:
            raise ValueError('Unsupported type of label matrix: ' + str(label_matrix.__type__))

        return statistics_factory_ptr.get().create()
