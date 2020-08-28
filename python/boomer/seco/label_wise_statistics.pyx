"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to store the elements of confusion matrices that are computed independently for
each label.
"""
from boomer.seco.label_wise_rule_evaluation cimport LabelWiseRuleEvaluation

from libcpp.memory cimport unique_ptr


cdef class LabelWiseStatisticsFactory(StatisticsFactory):
    """
    A wrapper for the C++ class `LabelWiseStatisticsFactoryImpl`.
    """

    def __cinit__(self, LabelWiseRuleEvaluation rule_evaluation):
        """
        :param rule_evaluation: The `LabelWiseRuleEvaluation` to be used for calculating the predictions, as well as
                                corresponding quality scores, of rules
        """
        self.rule_evaluation_ptr = rule_evaluation.rule_evaluation_ptr

    cdef AbstractStatistics* create(self, RandomAccessLabelMatrix label_matrix):
        cdef unique_ptr[AbstractStatisticsFactory] statistics_factory_ptr

        if isinstance(label_matrix, RandomAccessLabelMatrix):
            statistics_factory_ptr.reset(new LabelWiseStatisticsFactoryImpl(self.rule_evaluation_ptr,
                                                                            label_matrix.label_matrix_ptr))
        else:
            raise ValueError('Unsupported type of label matrix: ' + str(label_matrix.__type__))

        return statistics_factory_ptr.get().create()
