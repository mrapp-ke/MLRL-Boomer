"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to store gradients and Hessians that are calculated according to a
(decomposable) loss function that is applied label-wise.
"""
from boomer.common.input_data cimport RandomAccessLabelMatrix, AbstractLabelMatrix
from boomer.boosting.label_wise_losses cimport LabelWiseLoss
from boomer.boosting.label_wise_rule_evaluation cimport LabelWiseRuleEvaluation

from libcpp.memory cimport unique_ptr, dynamic_pointer_cast


cdef class LabelWiseStatisticsFactory(StatisticsFactory):
    """
    A wrapper for the C++ class `LabelWiseStatisticsFactoryImpl`.
    """

    def __cinit__(self, LabelWiseLoss loss_function, LabelWiseRuleEvaluation default_rule_evaluation,
                  LabelWiseRuleEvaluation rule_evaluation):
        """
        :param loss_function:           The loss function to be used for calculating gradients and Hessians
        :param default_rule_evaluation: The `LabelWiseRuleEvaluation` to be used for calculating the predictions, as
                                        well as corresponding quality scores, of the default rule
        :param rule_evaluation:         The `LabelWiseRuleEvaluation` to be used for calculating the predictions, as
                                        well as corresponding quality scores, of rules
        """
        self.loss_function_ptr = loss_function.loss_function_ptr
        self.default_rule_evaluation_ptr = default_rule_evaluation.rule_evaluation_ptr
        self.rule_evaluation_ptr = rule_evaluation.rule_evaluation_ptr

    cdef AbstractStatistics* create(self, LabelMatrix label_matrix):
        cdef unique_ptr[AbstractStatisticsFactory] statistics_factory_ptr

        if isinstance(label_matrix, RandomAccessLabelMatrix):
            statistics_factory_ptr.reset(new LabelWiseStatisticsFactoryImpl(
                self.loss_function_ptr, self.default_rule_evaluation_ptr, self.rule_evaluation_ptr,
                dynamic_pointer_cast[AbstractRandomAccessLabelMatrix, AbstractLabelMatrix](
                    label_matrix.label_matrix_ptr)))
        else:
            raise ValueError('Unsupported type of label matrix: ' + str(label_matrix.__type__))

        return statistics_factory_ptr.get().create()
