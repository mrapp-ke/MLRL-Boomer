"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
# pylint: disable=missing-function-docstring
from typing import Any, Optional, override

import pytest

from sklearn.utils.estimator_checks import check_estimator

from ..common.cmd_runner import CmdRunner
from ..common.datasets import Dataset
from ..common.integration_tests import RuleLearnerIntegrationTestsMixin
from ..common.integration_tests_classification import ClassificationIntegrationTests
from .cmd_builder import SeCoClassifierCmdBuilder

from mlrl.common.config.parameters import RulePruningParameter

from mlrl.seco.config.parameters import HEURISTIC_ACCURACY, HEURISTIC_F_MEASURE, HEURISTIC_LAPLACE, \
    HEURISTIC_M_ESTIMATE, HEURISTIC_PRECISION, HEURISTIC_RECALL, HEURISTIC_WRA, HeadTypeParameter, \
    LiftFunctionParameter
from mlrl.seco.learners import SeCoClassifier

from mlrl.util.cli import NONE


@pytest.mark.seco
@pytest.mark.classification
class TestSeCoClassifier(ClassificationIntegrationTests, RuleLearnerIntegrationTestsMixin):
    """
    Defines a series of integration tests for the separate-and-conquer (SeCo) algorithm for classification problems.
    """

    @override
    def _create_cmd_builder(self, dataset: str = Dataset.EMOTIONS) -> Any:
        return SeCoClassifierCmdBuilder(dataset=dataset)

    def test_scikit_learn_compatibility(self):
        check_estimator(SeCoClassifier())

    @pytest.mark.parametrize('heuristic', [
        HEURISTIC_ACCURACY,
        HEURISTIC_PRECISION,
        HEURISTIC_RECALL,
        HEURISTIC_LAPLACE,
        HEURISTIC_WRA,
        HEURISTIC_F_MEASURE,
        HEURISTIC_M_ESTIMATE,
    ])
    def test_heuristic(self, heuristic: str):
        builder = self._create_cmd_builder() \
            .heuristic(heuristic)
        CmdRunner(builder).run(f'heuristic_{heuristic}')

    @pytest.mark.parametrize('pruning_heuristic', [
        HEURISTIC_ACCURACY,
        HEURISTIC_PRECISION,
        HEURISTIC_RECALL,
        HEURISTIC_LAPLACE,
        HEURISTIC_WRA,
        HEURISTIC_F_MEASURE,
        HEURISTIC_M_ESTIMATE,
    ])
    def test_pruning_heuristic(self, pruning_heuristic: str):
        builder = self._create_cmd_builder() \
            .rule_pruning(RulePruningParameter.RULE_PRUNING_IREP) \
            .pruning_heuristic(pruning_heuristic)
        CmdRunner(builder).run(f'pruning-heuristic_{pruning_heuristic}')

    @pytest.mark.parametrize('head_type, lift_function', [
        (HeadTypeParameter.HEAD_TYPE_SINGLE, None),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL, NONE),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL, LiftFunctionParameter.LIFT_FUNCTION_PEAK),
        (HeadTypeParameter.HEAD_TYPE_PARTIAL, LiftFunctionParameter.LIFT_FUNCTION_KLN),
    ])
    def test_head_type(self, head_type: str, lift_function: Optional[str]):
        builder = self._create_cmd_builder() \
            .head_type(head_type) \
            .lift_function(lift_function) \
            .print_model_characteristics()
        CmdRunner(builder).run(f'head-type-{head_type}' + (f'_{lift_function}-lift-function' if lift_function else ''))
