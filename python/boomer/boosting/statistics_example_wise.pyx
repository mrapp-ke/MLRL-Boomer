"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to store gradients and Hessians that are calculated according to a
(non-decomposable) loss function that is applied example-wise.
"""
from boomer.common.input_data cimport RandomAccessLabelMatrix, ILabelMatrix

from libcpp.memory cimport make_shared, dynamic_pointer_cast
from libcpp.utility cimport move


cdef class ExampleWiseStatisticsFactory:
    """
    A wrapper for the pure virtual C++ class `IExampleWiseStatisticsFactory`.
    """

    cdef unique_ptr[IExampleWiseStatistics] create(self):
        """
        Creates a new instance of the class `IExampleWiseStatistics`.

        :return: An unique pointer to an object of type `IExampleWiseStatistics` that has been created
        """
        return self.statistics_factory_ptr.get().create()


cdef class DenseExampleWiseStatisticsFactory(ExampleWiseStatisticsFactory):
    """
    A wrapper for the C++ class `DenseExampleWiseStatisticsFactoryImpl`.
    """

    def __cinit__(self, ExampleWiseLoss loss_function, ExampleWiseRuleEvaluationFactory rule_evaluation_factory,
                 RandomAccessLabelMatrix label_matrix):
        """
        :param loss_function:           The loss function to be used for calculating gradients and Hessians
        :param rule_evaluation_factory: The `LabelWiseRuleEvaluation` that allows to create instances of the class that
                                        is used for calculating the predictions, as well as corresponding quality
                                        scores, of rules
        :param label_matrix:            A `RandomAccessLabelMatrix` that provides random access to the labels of the
                                        training examples
        """
        self.statistics_factory_ptr = <shared_ptr[IExampleWiseStatisticsFactory]>make_shared[DenseExampleWiseStatisticsFactoryImpl](
            loss_function.loss_function_ptr, rule_evaluation_factory.rule_evaluation_factory_ptr,
            dynamic_pointer_cast[IRandomAccessLabelMatrix, ILabelMatrix](label_matrix.label_matrix_ptr))


cdef class ExampleWiseStatisticsProvider(StatisticsProvider):
    """
    Provides access to an object of type `IExampleWiseStatistics`.
    """

    def __cinit__(self, ExampleWiseStatisticsFactory statistics_factory,
                  ExampleWiseRuleEvaluationFactory rule_evaluation_factory):
        """
        :param statistics_factory:      A factory that allows to create a new object of type `IExampleWiseStatistics`
        :param rule_evaluation_factory: The `ExampleWiseRuleEvaluationFactory` to switch to when invoking the function
                                        `switch_rule_evaluation`
        """
        cdef unique_ptr[IStatistics] statistics_ptr = <unique_ptr[IStatistics]>statistics_factory.create()
        self.statistics_ptr = <shared_ptr[IStatistics]>move(statistics_ptr)
        self.rule_evaluation_factory = rule_evaluation_factory

    cdef IStatistics* get(self):
        return self.statistics_ptr.get()

    cdef void switch_rule_evaluation(self):
        cdef ExampleWiseRuleEvaluationFactory rule_evaluation_factory = self.rule_evaluation_factory
        cdef shared_ptr[IExampleWiseRuleEvaluationFactory] rule_evaluation_factory_ptr = rule_evaluation_factory.rule_evaluation_factory_ptr
        dynamic_pointer_cast[IExampleWiseStatistics, IStatistics](self.statistics_ptr).get().setRuleEvaluationFactory(
            rule_evaluation_factory_ptr)


cdef class ExampleWiseStatisticsProviderFactory(StatisticsProviderFactory):
    """
    A factory that allows to create instances of the class `ExampleWiseStatisticsProvider`.
    """

    def __cinit__(self, ExampleWiseLoss loss_function, ExampleWiseRuleEvaluationFactory default_rule_evaluation_factory,
                  ExampleWiseRuleEvaluationFactory rule_evaluation_factory):
        """
        :param loss_function:                   The loss function to be used for calculating gradients and Hessians
        :param default_rule_evaluation_factory: The `ExampleWiseRuleEvaluation` to be used for calculating the
                                                predictions, as well as corresponding quality scores, of the default
                                                rule
        :param rule_evaluation_factory:         The `ExampleWiseRuleEvaluationFactory` to be used for calculating the
                                                predictions, as well as corresponding quality scores, of rules
        :param label_matrix:                    A label matrix that provides random access to the labels of the training
                                                examples
        """
        self.loss_function = loss_function
        self.default_rule_evaluation_factory = default_rule_evaluation_factory
        self.rule_evaluation_factory = rule_evaluation_factory

    cdef ExampleWiseStatisticsProvider create(self, LabelMatrix label_matrix):
        cdef ExampleWiseStatisticsFactory statistics_factory

        if isinstance(label_matrix, RandomAccessLabelMatrix):
            statistics_factory = DenseExampleWiseStatisticsFactory.__new__(DenseExampleWiseStatisticsFactory,
                                                                           self.loss_function,
                                                                           self.default_rule_evaluation_factory,
                                                                           label_matrix)
        else:
            raise ValueError('Unsupported type of label matrix: ' + str(label_matrix.__type__))

        return ExampleWiseStatisticsProvider.__new__(ExampleWiseStatisticsProvider, statistics_factory,
                                                     self.rule_evaluation_factory)
