"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from boomer.common.input cimport RandomAccessLabelMatrix, ILabelMatrix

from libcpp.memory cimport make_shared, dynamic_pointer_cast
from libcpp.utility cimport move


cdef class LabelWiseStatisticsFactory:
    """
    A wrapper for the pure virtual C++ class `ILabelWiseStatisticsFactory`.
    """

    cdef unique_ptr[ILabelWiseStatistics] create(self):
        """
        Creates a new instance of the type `ILabelWiseStatistics`.

        :return: An unique pointer to an object of type `ILabelWiseStatistics` that has been created
        """
        return self.statistics_factory_ptr.get().create()


cdef class DenseLabelWiseStatisticsFactory(LabelWiseStatisticsFactory):
    """
    A wrapper for the C++ class `DenseLabelWiseStatisticsFactory`.
    """

    def __cinit__(self, LabelWiseLoss loss_function, LabelWiseRuleEvaluationFactory rule_evaluation_factory,
                  RandomAccessLabelMatrix label_matrix):
        """
        :param loss_function:           The loss function to be used for calculating gradients and Hessians
        :param rule_evaluation_factory: The `LabelWiseRuleEvaluationFactory` that allows to create instances of the
                                        class that should be used for calculating the predictions, as well as
                                        corresponding quality scores, of rules
        :param label_matrix:            A `RandomAccessLabelMatrix` that provides random access to the labels of the
                                        training examples
        """
        self.statistics_factory_ptr = <shared_ptr[ILabelWiseStatisticsFactory]>make_shared[DenseLabelWiseStatisticsFactoryImpl](
            loss_function.loss_function_ptr, rule_evaluation_factory.rule_evaluation_factory_ptr,
            dynamic_pointer_cast[IRandomAccessLabelMatrix, ILabelMatrix](label_matrix.label_matrix_ptr))


cdef class LabelWiseStatisticsProvider(StatisticsProvider):
    """
    Provides access to an object of type `ILabelWiseStatistics`.
    """

    def __cinit__(self, LabelWiseStatisticsFactory statistics_factory,
                  LabelWiseRuleEvaluationFactory rule_evaluation_factory):
        """
        :param statistics_factory:      A factory that allows to create a new object of type `ILabelWiseStatistics`
        :param rule_evaluation_factory: The `LabelWiseRuleEvaluationFactory` to switch to when invoking the function
                                        `switch_rule_evaluation`
        """
        cdef unique_ptr[IStatistics] statistics_ptr = <unique_ptr[IStatistics]>statistics_factory.create()
        self.statistics_ptr = <shared_ptr[IStatistics]>move(statistics_ptr)
        self.rule_evaluation_factory = rule_evaluation_factory

    cdef IStatistics* get(self):
        return self.statistics_ptr.get()

    cdef void switch_rule_evaluation(self):
        cdef LabelWiseRuleEvaluationFactory rule_evaluation_factory = self.rule_evaluation_factory
        cdef shared_ptr[ILabelWiseRuleEvaluationFactory] rule_evaluation_factory_ptr = rule_evaluation_factory.rule_evaluation_factory_ptr
        dynamic_pointer_cast[ILabelWiseStatistics, IStatistics](self.statistics_ptr).get().setRuleEvaluationFactory(
            rule_evaluation_factory_ptr)


cdef class LabelWiseStatisticsProviderFactory(StatisticsProviderFactory):
    """
    A factory that allows to create instances of the class `LabelWiseStatisticsProvider`.
    """

    def __cinit__(self, LabelWiseLoss loss_function, LabelWiseRuleEvaluationFactory default_rule_evaluation_factory,
                  LabelWiseRuleEvaluationFactory rule_evaluation_factory):
        """
        :param loss_function:                   The loss function to be used for calculating gradients and Hessians
        :param default_rule_evaluation_factory: The `LabelWiseRuleEvaluationFactory` that allows to create instances of
                                                the class that should be used for calculating the predictions, as well
                                                as corresponding quality scores, of the default rule
        :param rule_evaluation:                 The `LabelWiseRuleEvaluationFactory` that allows to create instances of
                                                the class that should be used for calculating the predictions, as well
                                                as corresponding quality scores, of rules
        """
        self.loss_function = loss_function
        self.default_rule_evaluation_factory = default_rule_evaluation_factory
        self.rule_evaluation_factory = rule_evaluation_factory

    cdef LabelWiseStatisticsProvider create(self, LabelMatrix label_matrix):
        cdef LabelWiseStatisticsFactory statistics_factory

        if isinstance(label_matrix, RandomAccessLabelMatrix):
            statistics_factory = DenseLabelWiseStatisticsFactory.__new__(DenseLabelWiseStatisticsFactory,
                                                                         self.loss_function,
                                                                         self.default_rule_evaluation_factory,
                                                                         label_matrix)
        else:
            raise ValueError('Unsupported type of label matrix: ' + str(label_matrix.__type__))

        return LabelWiseStatisticsProvider.__new__(LabelWiseStatisticsProvider, statistics_factory,
                                                   self.rule_evaluation_factory)
