"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to store gradients and Hessians that are calculated according to a
(decomposable) loss function that is applied label-wise.
"""
from boomer.common.input_data cimport RandomAccessLabelMatrix
from boomer.boosting.label_wise_losses cimport LabelWiseLoss
from boomer.boosting.label_wise_rule_evaluation cimport LabelWiseRuleEvaluation

from libcpp.memory cimport make_shared


cdef class LabelWiseStatistics(GradientStatistics):
    """
    A wrapper for the C++ class `LabelWiseStatisticsImpl`.
    """

    def __cinit__(self, LabelWiseLoss loss_function, LabelWiseRuleEvaluation rule_evaluation):
        """
        :param loss_function:   The loss function to be used for calculating gradients and Hessians
        :param rule_evaluation: The `LabelWiseRuleEvaluation` to be used for calculating the predictions, as well as
                                corresponding quality scores, of rules
        """
        cdef shared_ptr[AbstractLabelWiseLoss] loss_function_ptr = loss_function.loss_function_ptr
        cdef shared_ptr[LabelWiseRuleEvaluationImpl] rule_evaluation_ptr = rule_evaluation.rule_evaluation_ptr
        self.statistics_ptr = <shared_ptr[AbstractStatistics]>make_shared[LabelWiseStatisticsImpl](loss_function_ptr,
                                                                                                   rule_evaluation_ptr)


cdef class LabelWiseStatisticsFactory(StatisticsFactory):
    """
    A wrapper for the C++ class `LabelWiseStatisticsFactoryImpl`.
    """

    def __cinit__(self, LabelWiseLoss loss_function, LabelWiseRuleEvaluation rule_evaluation,
                  RandomAccessLabelMatrix label_matrix):
        """
        :param loss_function:   The loss function to be used for calculating gradients and Hessians
        :param rule_evaluation: The `LabelWiseRuleEvaluation` to be used for calculating the predictions, as well as
                                corresponding quality scores, of rules
        :param label_matrix:    A label matrix that provides random access to the labels of the training examples
        """
        cdef shared_ptr[AbstractLabelWiseLoss] loss_function_ptr = loss_function.loss_function_ptr
        cdef shared_ptr[LabelWiseRuleEvaluationImpl] rule_evaluation_ptr = rule_evaluation.rule_evaluation_ptr

        if isinstance(label_matrix, RandomAccessLabelMatrix):
            self.statistics_factory_ptr =  <shared_ptr[AbstractStatisticsFactory]>make_shared[LabelWiseStatisticsFactoryImpl](
                loss_function_ptr, rule_evaluation_ptr, label_matrix.label_matrix_ptr)
        else:
            raise ValueError('Unsupported type of label matrix: ' + str(label_matrix.__type__))
