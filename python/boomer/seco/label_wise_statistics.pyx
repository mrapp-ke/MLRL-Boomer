"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to store the elements of confusion matrices that are computed independently for
each label.
"""
from boomer.common.input_data cimport RandomAccessLabelMatrix, ILabelMatrix

from libcpp.memory cimport make_shared, dynamic_pointer_cast
from libcpp.utility cimport move


cdef class LabelWiseStatisticsFactory:
    """
    A wrapper for the pure virtual C++ class `ILabelWiseStatisticsFactory`.
    """

    cdef unique_ptr[AbstractLabelWiseStatistics] create(self):
        """
        Creates a new instance of the class `AbstractLabelWiseStatistics`.

        :return: An unique pointer to an object of type `AbstractLabelWiseStatistics` that has been created
        """
        return self.statistics_factory_ptr.get().create()


cdef class DenseLabelWiseStatisticsFactory(LabelWiseStatisticsFactory):
    """
    A wrapper for the C++ class `DenseLabelWiseStatisticsFactoryImpl`.
    """

    def __cinit__(self, LabelWiseRuleEvaluationFactory rule_evaluation_factory, RandomAccessLabelMatrix label_matrix):
        """
        :param rule_evaluation_factory: The `LabelWiseRuleEvaluationFactory` that allows to create instances of the
                                        class that should be used for calculating the predictions, as well as
                                        corresponding quality scores, of rules
        :param label_matrix:            A `RandomAccessLabelMatrix` that provides random access to the labels of the
                                        training examples
        """
        self.statistics_factory_ptr = <shared_ptr[ILabelWiseStatisticsFactory]>make_shared[DenseLabelWiseStatisticsFactoryImpl](
            rule_evaluation_factory.rule_evaluation_factory_ptr,
            dynamic_pointer_cast[IRandomAccessLabelMatrix, ILabelMatrix](label_matrix.label_matrix_ptr))


cdef class LabelWiseStatisticsProvider(StatisticsProvider):
    """
    Provides access to an object of type `AbstractLabelWiseStatistics`.
    """

    def __cinit__(self, LabelWiseStatisticsFactory statistics_factory,
                  LabelWiseRuleEvaluationFactory rule_evaluation_factory):
        """
        :param statistics_factory:      A factory that allows to create a new object of type `AbstractLabelWiseStatistics`
        :param rule_evaluation_factory: The `LabelWiseRuleEvaluationFactory` to switch to when invoking the function
                                        `switch_rule_evaluation`
        """
        cdef unique_ptr[AbstractStatistics] statistics_ptr = <unique_ptr[AbstractStatistics]>statistics_factory.create()
        self.statistics_ptr = <shared_ptr[AbstractStatistics]>move(statistics_ptr)
        self.rule_evaluation_factory = rule_evaluation_factory

    cdef AbstractStatistics* get(self):
        return self.statistics_ptr.get()

    cdef void switch_rule_evaluation(self):
        cdef LabelWiseRuleEvaluationFactory rule_evaluation_factory = self.rule_evaluation_factory
        cdef shared_ptr[ILabelWiseRuleEvaluationFactory] rule_evaluation_factory_ptr = rule_evaluation_factory.rule_evaluation_factory_ptr
        dynamic_pointer_cast[AbstractLabelWiseStatistics, AbstractStatistics](self.statistics_ptr).get().setRuleEvaluationFactory(
            rule_evaluation_factory_ptr)


cdef class LabelWiseStatisticsProviderFactory(StatisticsProviderFactory):
    """
    A factory that allows to create instances of the class `LabelWiseStatisticsProvider`.
    """

    def __cinit__(self, LabelWiseRuleEvaluationFactory default_rule_evaluation_factory,
                  LabelWiseRuleEvaluationFactory rule_evaluation_factory):
        """
        :param default_rule_evaluation_factory: The `LabelWiseRuleEvaluationFactory` that allows to create instances of
                                                the class that should be used for calculating the predictions, as well
                                                as corresponding quality scores, of the default rule
        :param rule_evaluation_factory:         The `LabelWiseRuleEvaluation` that allows to create instances of the
                                                class that should be used for calculating the predictions, as well as
                                                corresponding quality scores, of rules
        """
        self.default_rule_evaluation_factory = default_rule_evaluation_factory
        self.rule_evaluation_factory = rule_evaluation_factory

    cdef LabelWiseStatisticsProvider create(self, LabelMatrix label_matrix):
        cdef LabelWiseStatisticsFactory statistics_factory

        if isinstance(label_matrix, RandomAccessLabelMatrix):
            statistics_factory = DenseLabelWiseStatisticsFactory.__new__(DenseLabelWiseStatisticsFactory,
                                                                         self.default_rule_evaluation_factory,
                                                                         label_matrix)
        else:
            raise ValueError('Unsupported type of label matrix: ' + str(label_matrix.__type__))

        return LabelWiseStatisticsProvider.__new__(LabelWiseStatisticsProvider, statistics_factory,
                                                   self.rule_evaluation_factory)
