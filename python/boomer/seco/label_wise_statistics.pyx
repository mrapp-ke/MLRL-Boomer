"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to store the elements of confusion matrices that are computed independently for
each label.
"""
from boomer.common.input_data cimport RandomAccessLabelMatrix, AbstractLabelMatrix
from boomer.seco.label_wise_rule_evaluation cimport LabelWiseRuleEvaluation

from libcpp.memory cimport unique_ptr, dynamic_pointer_cast


cdef class LabelWiseStatisticsFactory(StatisticsFactory):
    """
    A factory that allows to create instances of the class `LabelWiseStatisticsImpl`.
    """

    def __cinit__(self, LabelWiseRuleEvaluation default_rule_evaluation, LabelWiseRuleEvaluation rule_evaluation):
        """
        :param default_rule_evaluation: The `LabelWiseRuleEvaluation` to be used for calculating the predictions, as
                                        well as corresponding quality scores, of the default rule
        :param rule_evaluation:         The `LabelWiseRuleEvaluation` to be used for calculating the predictions, as
                                        well as corresponding quality scores, of rules
        """
        self.default_rule_evaluation_ptr = default_rule_evaluation.rule_evaluation_ptr
        self.rule_evaluation_ptr = rule_evaluation.rule_evaluation_ptr

    cdef AbstractStatistics* create_initial_statistics(self, LabelMatrix label_matrix):
        cdef unique_ptr[AbstractStatisticsFactory] statistics_factory_ptr

        if isinstance(label_matrix, RandomAccessLabelMatrix):
            statistics_factory_ptr.reset(new LabelWiseStatisticsFactoryImpl(
                self.default_rule_evaluation_ptr,
                dynamic_pointer_cast[AbstractRandomAccessLabelMatrix, AbstractLabelMatrix](
                    label_matrix.label_matrix_ptr)))
        else:
            raise ValueError('Unsupported type of label matrix: ' + str(label_matrix.__type__))

        return statistics_factory_ptr.get().create()

    cdef AbstractStatistics* copy_statistics(self, AbstractStatistics* statistics):
        return statistics
