"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to store the elements of confusion matrices that are computed independently for
each label.
"""
from boomer.common.input_data cimport RandomAccessLabelMatrix, AbstractLabelMatrix
from boomer.seco.label_wise_rule_evaluation cimport LabelWiseRuleEvaluation

from libcpp.memory cimport make_shared, dynamic_pointer_cast


cdef class LabelWiseStatisticsFactory:
    """
    A wrapper for the abstract C++ class `AbstractLabelWiseStatisticsFactory`.
    """

    cdef AbstractLabelWiseStatistics* create(self):
        """
        Creates a new instance of the class `AbstractLabelWiseStatistics`.

        :return: A pointer to an object of type `AbstractLabelWiseStatistics` that has been created
        """
        return self.statistics_factory_ptr.get().create()


cdef class DenseLabelWiseStatisticsFactory(LabelWiseStatisticsFactory):
    """
    A wrapper for the C++ class `DenseLabelWiseStatisticsFactoryImpl`.
    """

    def __cinit__(self, LabelWiseRuleEvaluation rule_evaluation, RandomAccessLabelMatrix label_matrix):
        """
        :param rule_evaluation: The `LabelWiseRuleEvaluation` to be used for calculating the predictions, as well as
                                corresponding quality scores, of rules
        :param label_matrix:    A `RandomAccessLabelMatrix` that provides random access to the labels of the training
                                examples
        """
        self.statistics_factory_ptr = <shared_ptr[AbstractLabelWiseStatisticsFactory]>make_shared[DenseLabelWiseStatisticsFactoryImpl](
            rule_evaluation.rule_evaluation_ptr,
            dynamic_pointer_cast[AbstractRandomAccessLabelMatrix, AbstractLabelMatrix](label_matrix.label_matrix_ptr))


cdef class LabelWiseStatisticsProvider(StatisticsProvider):
    """
    Provides access to an object of type `AbstractLabelWiseStatistics`.
    """

    def __cinit__(self, LabelWiseStatisticsFactory statistics_factory, LabelWiseRuleEvaluation rule_evaluation):
        """
        :param statistics_factory:  A factory that allows to create a new object of type `AbstractLabelWiseStatistics`
        :param rule_evaluation:     The `LabelWiseRuleEvaluation` to switch to when invoking the function
                                    `switch_rule_evaluation`
        """
        self.statistics_ptr = shared_ptr[AbstractLabelWiseStatistics](statistics_factory.create())
        self.rule_evaluation = rule_evaluation

    cdef AbstractStatistics* get(self):
        return self.statistics_ptr.get()


cdef class LabelWiseStatisticsProviderFactory(StatisticsProviderFactory):
    """
    A factory that allows to create instances of the class `LabelWiseStatisticsProvider`.
    """

    def __cinit__(self, LabelWiseRuleEvaluation default_rule_evaluation, LabelWiseRuleEvaluation rule_evaluation):
        """
        :param default_rule_evaluation: The `LabelWiseRuleEvaluation` to be used for calculating the predictions, as
                                        well as corresponding quality scores, of the default rule
        :param rule_evaluation:         The `LabelWiseRuleEvaluation` to be used for calculating the predictions, as
                                        well as corresponding quality scores, of rules
        """
        self.default_rule_evaluation = default_rule_evaluation
        self.rule_evaluation = rule_evaluation

    cdef LabelWiseStatisticsProvider create(self, LabelMatrix label_matrix):
        cdef LabelWiseStatisticsFactory statistics_factory

        if isinstance(label_matrix, RandomAccessLabelMatrix):
            statistics_factory = DenseLabelWiseStatisticsFactory.__new__(DenseLabelWiseStatisticsFactory,
                                                                         self.rule_evaluation, label_matrix)
        else:
            raise ValueError('Unsupported type of label matrix: ' + str(label_matrix.__type__))

        return LabelWiseStatisticsProvider.__new__(LabelWiseStatisticsProvider, statistics_factory,
                                                   self.rule_evaluation)
