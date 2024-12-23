"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from typing import Any

from ..common.cmd_builder import DATASET_EMOTIONS, RULE_PRUNING_IREP
from ..common.integration_tests_classification import ClassificationIntegrationTests
from .cmd_builder import HEAD_TYPE_PARTIAL, HEAD_TYPE_SINGLE, HEURISTIC_ACCURACY, HEURISTIC_F_MEASURE, \
    HEURISTIC_LAPLACE, HEURISTIC_M_ESTIMATE, HEURISTIC_PRECISION, HEURISTIC_RECALL, HEURISTIC_WRA, LIFT_FUNCTION_KLN, \
    LIFT_FUNCTION_NO, LIFT_FUNCTION_PEAK, SeCoClassifierCmdBuilder


class SeCoClassifierIntegrationTests(ClassificationIntegrationTests):
    """
    Defines a series of integration tests for the separate-and-conquer (SeCo) algorithm for classification problems.
    """

    def __init__(self, methodName='runTest'):
        super().__init__(methodName=methodName)

    def _create_cmd_builder(self, dataset: str = DATASET_EMOTIONS) -> Any:
        return SeCoClassifierCmdBuilder(self, dataset=dataset)

    def test_heuristic_accuracy(self):
        """
        Tests the SeCo algorithm when using the accuracy heuristic for learning rules.
        """
        builder = self._create_cmd_builder() \
            .heuristic(HEURISTIC_ACCURACY)
        builder.run_cmd('heuristic_accuracy')

    def test_heuristic_precision(self):
        """
        Tests the SeCo algorithm when using the precision heuristic for learning rules.
        """
        builder = self._create_cmd_builder() \
            .heuristic(HEURISTIC_PRECISION)
        builder.run_cmd('heuristic_precision')

    def test_heuristic_recall(self):
        """
        Tests the SeCo algorithm when using the recall heuristic for learning rules.
        """
        builder = self._create_cmd_builder() \
            .heuristic(HEURISTIC_RECALL)
        builder.run_cmd('heuristic_recall')

    def test_heuristic_laplace(self):
        """
        Tests the SeCo algorithm when using the Laplace heuristic for learning rules.
        """
        builder = self._create_cmd_builder() \
            .heuristic(HEURISTIC_LAPLACE)
        builder.run_cmd('heuristic_laplace')

    def test_heuristic_wra(self):
        """
        Tests the SeCo algorithm when using the WRA heuristic for learning rules.
        """
        builder = self._create_cmd_builder() \
            .heuristic(HEURISTIC_WRA)
        builder.run_cmd('heuristic_wra')

    def test_heuristic_f_measure(self):
        """
        Tests the SeCo algorithm when using the F-measure heuristic for learning rules.
        """
        builder = self._create_cmd_builder() \
            .heuristic(HEURISTIC_F_MEASURE)
        builder.run_cmd('heuristic_f-measure')

    def test_heuristic_m_estimate(self):
        """
        Tests the SeCo algorithm when using the m-estimate heuristic for learning rules.
        """
        builder = self._create_cmd_builder() \
            .heuristic(HEURISTIC_M_ESTIMATE)
        builder.run_cmd('heuristic_m-estimate')

    def test_pruning_heuristic_accuracy(self):
        """
        Tests the SeCo algorithm when using the accuracy heuristic for pruning rules.
        """
        builder = self._create_cmd_builder() \
            .rule_pruning(RULE_PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_ACCURACY)
        builder.run_cmd('pruning-heuristic_accuracy')

    def test_pruning_heuristic_precision(self):
        """
        Tests the SeCo algorithm when using the precision heuristic for pruning rules.
        """
        builder = self._create_cmd_builder() \
            .rule_pruning(RULE_PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_PRECISION)
        builder.run_cmd('pruning-heuristic_precision')

    def test_pruning_heuristic_recall(self):
        """
        Tests the SeCo algorithm when using the recall heuristic for pruning rules.
        """
        builder = self._create_cmd_builder() \
            .rule_pruning(RULE_PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_RECALL)
        builder.run_cmd('pruning-heuristic_recall')

    def test_pruning_heuristic_laplace(self):
        """
        Tests the SeCo algorithm when using the Laplace heuristic for pruning rules.
        """
        builder = self._create_cmd_builder() \
            .rule_pruning(RULE_PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_LAPLACE)
        builder.run_cmd('pruning-heuristic_laplace')

    def test_pruning_heuristic_wra(self):
        """
        Tests the SeCo algorithm when using the WRA heuristic for pruning rules.
        """
        builder = self._create_cmd_builder() \
            .rule_pruning(RULE_PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_WRA)
        builder.run_cmd('pruning-heuristic_wra')

    def test_pruning_heuristic_f_measure(self):
        """
        Tests the SeCo algorithm when using the F-measure heuristic for pruning rules.
        """
        builder = self._create_cmd_builder() \
            .rule_pruning(RULE_PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_F_MEASURE)
        builder.run_cmd('pruning-heuristic_f-measure')

    def test_pruning_heuristic_m_estimate(self):
        """
        Tests the SeCo algorithm when using the m-estimate heuristic for pruning rules.
        """
        builder = self._create_cmd_builder() \
            .rule_pruning(RULE_PRUNING_IREP) \
            .pruning_heuristic(HEURISTIC_M_ESTIMATE)
        builder.run_cmd('pruning-heuristic_m-estimate')

    def test_single_output_heads(self):
        """
        Tests the SeCo algorithm when inducing rules with single-output heads.
        """
        builder = self._create_cmd_builder() \
            .head_type(HEAD_TYPE_SINGLE) \
            .print_model_characteristics()
        builder.run_cmd('single-output-heads')

    def test_partial_heads_no_lift_function(self):
        """
        Tests the SeCo algorithm when inducing partial rules using no lift function.
        """
        builder = self._create_cmd_builder() \
            .head_type(HEAD_TYPE_PARTIAL) \
            .lift_function(LIFT_FUNCTION_NO) \
            .print_model_characteristics()
        builder.run_cmd('partial-heads_no-lift-function')

    def test_partial_heads_peak_lift_function(self):
        """
        Tests the SeCo algorithm when inducing partial rules using the peak lift function.
        """
        builder = self._create_cmd_builder() \
            .head_type(HEAD_TYPE_PARTIAL) \
            .lift_function(LIFT_FUNCTION_PEAK) \
            .print_model_characteristics()
        builder.run_cmd('partial-heads_peak-lift-function')

    def test_partial_heads_kln_lift_function(self):
        """
        Tests the SeCo algorithm when inducing partial rules using the KLN lift function.
        """
        builder = self._create_cmd_builder() \
            .head_type(HEAD_TYPE_PARTIAL) \
            .lift_function(LIFT_FUNCTION_KLN) \
            .print_model_characteristics()
        builder.run_cmd('partial-heads_kln-lift-function')
