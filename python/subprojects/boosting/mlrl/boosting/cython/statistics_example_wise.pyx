"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._types cimport uint32
from mlrl.common.cython.measures cimport EvaluationMeasureFactory
from mlrl.boosting.cython.losses_example_wise cimport ExampleWiseLossFactory
from mlrl.boosting.cython.rule_evaluation_example_wise cimport ExampleWiseRuleEvaluationFactory
from mlrl.boosting.cython.rule_evaluation_label_wise cimport LabelWiseRuleEvaluationFactory

from libcpp.utility cimport move
from libcpp.memory cimport unique_ptr, make_unique


cdef class DenseExampleWiseStatisticsProviderFactory(StatisticsProviderFactory):
    """
    A wrapper for the C++ class `DenseExampleWiseStatisticsProviderFactory`.
    """

    def __cinit__(self, ExampleWiseLossFactory loss_factory not None,
                  EvaluationMeasureFactory evaluation_measure_factory not None,
                  ExampleWiseRuleEvaluationFactory default_rule_evaluation_factory not None,
                  ExampleWiseRuleEvaluationFactory regular_rule_evaluation_factory not None,
                  ExampleWiseRuleEvaluationFactory pruning_rule_evaluation_factory not None, uint32 num_threads):
        """
        :param loss_factory:                    The `ExampleWiseLossFactory` to be used for calculating gradients and
                                                Hessians
        :param evaluation_measure_factory:      The `EvaluationMeasureFactory` to be used for assessing the quality of
                                                predictions
        :param default_rule_evaluation_factory: The `ExampleWiseRuleEvaluation` to be used for calculating the
                                                predictions, as well as corresponding quality scores, of the default
                                                rule
        :param regular_rule_evaluation_factory: The `ExampleWiseRuleEvaluationFactory` to be used for calculating the
                                                predictions, as well as corresponding quality scores, of all remaining
                                                rules
        :param pruning_rule_evaluation_factory: The `ExampleWiseRuleEvaluationFactory` to be used for calculating the
                                                predictions, as well as corresponding quality scores, when pruning rules
        :param label_matrix:                    A label matrix that provides random access to the labels of the training
                                                examples
        :param num_threads:                     The number of CPU threads to be used to calculate the initial statistics
                                                in parallel. Must be at least 1
        """
        self.statistics_provider_factory_ptr = <unique_ptr[IStatisticsProviderFactory]>make_unique[DenseExampleWiseStatisticsProviderFactoryImpl](
            move(loss_factory.loss_factory_ptr), move(evaluation_measure_factory.get_evaluation_measure_factory_ptr()),
            move(default_rule_evaluation_factory.rule_evaluation_factory_ptr),
            move(regular_rule_evaluation_factory.rule_evaluation_factory_ptr),
            move(pruning_rule_evaluation_factory.rule_evaluation_factory_ptr), num_threads)


cdef class DenseConvertibleExampleWiseStatisticsProviderFactory(StatisticsProviderFactory):
    """
    A wrapper for the C++ class `DenseConvertibleExampleWiseStatisticsProviderFactory`.
    """

    def __cinit__(self, ExampleWiseLossFactory loss_factory not None,
                  EvaluationMeasureFactory evaluation_measure_factory not None,
                  ExampleWiseRuleEvaluationFactory default_rule_evaluation_factory not None,
                  LabelWiseRuleEvaluationFactory regular_rule_evaluation_factory not None,
                  LabelWiseRuleEvaluationFactory pruning_rule_evaluation_factory not None, uint32 num_threads):
        """
        :param loss_factory:                    The `ExampleWiseLossFactory` to be used for calculating gradients and
                                                Hessians
        :param evaluation_measure_factory:      The `EvaluationMeasureFactory` to be used for assessing the quality of
                                                predictions
        :param default_rule_evaluation_factory: The `ExampleWiseRuleEvaluation` to be used for calculating the
                                                predictions, as well as corresponding quality scores, of the default
                                                rule
        :param regular_rule_evaluation_factory: The `LabelWiseRuleEvaluationFactory` to be used for calculating the
                                                predictions, as well as corresponding quality scores, of all remaining
                                                rules
        :param pruning_rule_evaluation_factory: The `LabelWiseRuleEvaluationFactory` to be used for calculating the
                                                predictions, as well as corresponding quality scores, when pruning rules
        :param label_matrix:                    A label matrix that provides random access to the labels of the training
                                                examples
        :param num_threads:                     The number of CPU threads to be used to calculate the initial statistics
                                                in parallel. Must be at least 1
        """
        self.statistics_provider_factory_ptr = <unique_ptr[IStatisticsProviderFactory]>make_unique[DenseConvertibleExampleWiseStatisticsProviderFactoryImpl](
            move(loss_factory.loss_factory_ptr), move(evaluation_measure_factory.get_evaluation_measure_factory_ptr()),
            move(default_rule_evaluation_factory.rule_evaluation_factory_ptr),
            move(regular_rule_evaluation_factory.rule_evaluation_factory_ptr),
            move(pruning_rule_evaluation_factory.rule_evaluation_factory_ptr), num_threads)
