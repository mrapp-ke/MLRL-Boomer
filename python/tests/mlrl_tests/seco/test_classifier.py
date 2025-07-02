"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from typing import Any

from ..common.cmd_builder import CmdBuilder
from ..common.cmd_runner import CmdRunner
from ..common.datasets import Dataset
from ..common.integration_tests_classification import ClassificationIntegrationTests
from .cmd_builder import SeCoClassifierCmdBuilder


class SeCoClassifierIntegrationTests(ClassificationIntegrationTests):
    """
    Defines a series of integration tests for the separate-and-conquer (SeCo) algorithm for classification problems.
    """

    # pylint: disable=invalid-name
    def __init__(self, methodName='runTest'):
        super().__init__(methodName=methodName)

    def _create_cmd_builder(self, dataset: str = Dataset.EMOTIONS) -> Any:
        return SeCoClassifierCmdBuilder(dataset=dataset)

    def test_heuristic_accuracy(self):
        builder = self._create_cmd_builder() \
            .heuristic(SeCoClassifierCmdBuilder.HEURISTIC_ACCURACY)
        CmdRunner(self, builder).run('heuristic_accuracy')

    def test_heuristic_precision(self):
        builder = self._create_cmd_builder() \
            .heuristic(SeCoClassifierCmdBuilder.HEURISTIC_PRECISION)
        CmdRunner(self, builder).run('heuristic_precision')

    def test_heuristic_recall(self):
        builder = self._create_cmd_builder() \
            .heuristic(SeCoClassifierCmdBuilder.HEURISTIC_RECALL)
        CmdRunner(self, builder).run('heuristic_recall')

    def test_heuristic_laplace(self):
        builder = self._create_cmd_builder() \
            .heuristic(SeCoClassifierCmdBuilder.HEURISTIC_LAPLACE)
        CmdRunner(self, builder).run('heuristic_laplace')

    def test_heuristic_wra(self):
        builder = self._create_cmd_builder() \
            .heuristic(SeCoClassifierCmdBuilder.HEURISTIC_WRA)
        CmdRunner(self, builder).run('heuristic_wra')

    def test_heuristic_f_measure(self):
        builder = self._create_cmd_builder() \
            .heuristic(SeCoClassifierCmdBuilder.HEURISTIC_F_MEASURE)
        CmdRunner(self, builder).run('heuristic_f-measure')

    def test_heuristic_m_estimate(self):
        builder = self._create_cmd_builder() \
            .heuristic(SeCoClassifierCmdBuilder.HEURISTIC_M_ESTIMATE)
        CmdRunner(self, builder).run('heuristic_m-estimate')

    def test_pruning_heuristic_accuracy(self):
        builder = self._create_cmd_builder() \
            .rule_pruning(CmdBuilder.RULE_PRUNING_IREP) \
            .pruning_heuristic(SeCoClassifierCmdBuilder.HEURISTIC_ACCURACY)
        CmdRunner(self, builder).run('pruning-heuristic_accuracy')

    def test_pruning_heuristic_precision(self):
        builder = self._create_cmd_builder() \
            .rule_pruning(CmdBuilder.RULE_PRUNING_IREP) \
            .pruning_heuristic(SeCoClassifierCmdBuilder.HEURISTIC_PRECISION)
        CmdRunner(self, builder).run('pruning-heuristic_precision')

    def test_pruning_heuristic_recall(self):
        builder = self._create_cmd_builder() \
            .rule_pruning(CmdBuilder.RULE_PRUNING_IREP) \
            .pruning_heuristic(SeCoClassifierCmdBuilder.HEURISTIC_RECALL)
        CmdRunner(self, builder).run('pruning-heuristic_recall')

    def test_pruning_heuristic_laplace(self):
        builder = self._create_cmd_builder() \
            .rule_pruning(CmdBuilder.RULE_PRUNING_IREP) \
            .pruning_heuristic(SeCoClassifierCmdBuilder.HEURISTIC_LAPLACE)
        CmdRunner(self, builder).run('pruning-heuristic_laplace')

    def test_pruning_heuristic_wra(self):
        builder = self._create_cmd_builder() \
            .rule_pruning(CmdBuilder.RULE_PRUNING_IREP) \
            .pruning_heuristic(SeCoClassifierCmdBuilder.HEURISTIC_WRA)
        CmdRunner(self, builder).run('pruning-heuristic_wra')

    def test_pruning_heuristic_f_measure(self):
        builder = self._create_cmd_builder() \
            .rule_pruning(CmdBuilder.RULE_PRUNING_IREP) \
            .pruning_heuristic(SeCoClassifierCmdBuilder.HEURISTIC_F_MEASURE)
        CmdRunner(self, builder).run('pruning-heuristic_f-measure')

    def test_pruning_heuristic_m_estimate(self):
        builder = self._create_cmd_builder() \
            .rule_pruning(CmdBuilder.RULE_PRUNING_IREP) \
            .pruning_heuristic(SeCoClassifierCmdBuilder.HEURISTIC_M_ESTIMATE)
        CmdRunner(self, builder).run('pruning-heuristic_m-estimate')

    def test_single_output_heads(self):
        builder = self._create_cmd_builder() \
            .head_type(SeCoClassifierCmdBuilder.HEAD_TYPE_SINGLE) \
            .print_model_characteristics()
        CmdRunner(self, builder).run('single-output-heads')

    def test_partial_heads_no_lift_function(self):
        builder = self._create_cmd_builder() \
            .head_type(SeCoClassifierCmdBuilder.HEAD_TYPE_PARTIAL) \
            .lift_function(SeCoClassifierCmdBuilder.LIFT_FUNCTION_NO) \
            .print_model_characteristics()
        CmdRunner(self, builder).run('partial-heads_no-lift-function')

    def test_partial_heads_peak_lift_function(self):
        builder = self._create_cmd_builder() \
            .head_type(SeCoClassifierCmdBuilder.HEAD_TYPE_PARTIAL) \
            .lift_function(SeCoClassifierCmdBuilder.LIFT_FUNCTION_PEAK) \
            .print_model_characteristics()
        CmdRunner(self, builder).run('partial-heads_peak-lift-function')

    def test_partial_heads_kln_lift_function(self):
        builder = self._create_cmd_builder() \
            .head_type(SeCoClassifierCmdBuilder.HEAD_TYPE_PARTIAL) \
            .lift_function(SeCoClassifierCmdBuilder.LIFT_FUNCTION_KLN) \
            .print_model_characteristics()
        CmdRunner(self, builder).run('partial-heads_kln-lift-function')
