"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to store the elements of confusion matrices that are computed independently for
each label.
"""
from boomer.seco.label_wise_rule_evaluation cimport LabelWiseRuleEvaluation

from libcpp.memory cimport make_shared


cdef class LabelWiseStatistics(CoverageStatistics):
    """
    A wrapper for the C++ class `LabelWiseStatisticsImpl`.
    """

    def __cinit__(self, LabelWiseRuleEvaluation rule_evaluation):
        """
        :param rule_evaluation: The `LabelWiseRuleEvaluation` to be used for calculating the predictions, as well as
                                corresponding quality scores, of rules
        """
        cdef shared_ptr[LabelWiseRuleEvaluationImpl] rule_evaluation_ptr = rule_evaluation.rule_evaluation_ptr
        self.statistics_ptr = <shared_ptr[AbstractStatistics]>make_shared[LabelWiseStatisticsImpl](rule_evaluation_ptr)


cdef class LabelWiseStatisticsFactory(StatisticsFactory):
    """
    A wrapper for the C++ class `LabelWiseStatisticsFactoryImpl`.
    """

    def __cinit__(self, LabelWiseRuleEvaluation rule_evaluation, RandomAccessLabelMatrix label_matrix):
        """
        :param rule_evaluation: The `LabelWiseRuleEvaluation` to be used for calculating the predictions, as well as
                                corresponding quality scores, of rules
        :param label_matrix:    A label matrix that provides random access to the labels of the training examples
        """
        cdef shared_ptr[LabelWiseRuleEvaluationImpl] rule_evaluation_ptr = rule_evaluation.rule_evaluation_ptr

        if isinstance(label_matrix, RandomAccessLabelMatrix):
            self.statistics_factory_ptr =  <shared_ptr[AbstractStatisticsFactory]>make_shared[LabelWiseStatisticsFactoryImpl](
                rule_evaluation_ptr, label_matrix.label_matrix_ptr)
        else:
            raise ValueError('Unsupported type of label matrix: ' + str(label_matrix.__type__))
