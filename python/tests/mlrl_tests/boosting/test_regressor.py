"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
# pylint: disable=missing-function-docstring
from typing import Any, override

import pytest

from ..common.cmd_runner import CmdRunner
from ..common.datasets import Dataset
from ..common.integration_tests_regression import RegressionIntegrationTests
from .cmd_builder_regression import BoomerRegressorCmdBuilder
from .integration_tests import BoomerIntegrationTestsMixin

from mlrl.boosting.config.parameters import HeadTypeParameter, RegressionLossParameter, StatisticTypeParameter


@pytest.mark.boosting
@pytest.mark.regression
class TestBoomerRegressor(RegressionIntegrationTests, BoomerIntegrationTestsMixin):
    """
    Defines a series of integration tests for the BOOMER algorithm for regression problems.
    """

    @override
    def _create_cmd_builder(self, dataset: str = Dataset.ATP7D) -> Any:
        return BoomerRegressorCmdBuilder(dataset=dataset)

    @pytest.mark.parametrize('head_type', [
        HeadTypeParameter.HEAD_TYPE_SINGLE,
        HeadTypeParameter.HEAD_TYPE_COMPLETE,
        HeadTypeParameter.HEAD_TYPE_PARTIAL_FIXED,
        HeadTypeParameter.HEAD_TYPE_PARTIAL_DYNAMIC,
    ])
    @pytest.mark.parametrize('statistic_type', [
        StatisticTypeParameter.STATISTIC_TYPE_FLOAT32,
        StatisticTypeParameter.STATISTIC_TYPE_FLOAT64,
    ])
    def test_decomposable_head_type(self, head_type: str, statistic_type: str):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_DECOMPOSABLE) \
            .statistic_type(statistic_type) \
            .head_type(head_type) \
            .print_model_characteristics()
        CmdRunner(builder).run(f'decomposable-{head_type}-heads_{statistic_type}-statistics')

    @pytest.mark.parametrize('head_type', [
        HeadTypeParameter.HEAD_TYPE_SINGLE,
        pytest.param(HeadTypeParameter.HEAD_TYPE_COMPLETE, marks=pytest.mark.flaky(reruns=5)),
        HeadTypeParameter.HEAD_TYPE_PARTIAL_FIXED,
        pytest.param(HeadTypeParameter.HEAD_TYPE_PARTIAL_DYNAMIC, marks=pytest.mark.flaky(reruns=5)),
    ])
    @pytest.mark.parametrize('statistic_type', [
        StatisticTypeParameter.STATISTIC_TYPE_FLOAT32,
        StatisticTypeParameter.STATISTIC_TYPE_FLOAT64,
    ])
    def test_non_decomposable_head_type(self, head_type: str, statistic_type: str):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_NON_DECOMPOSABLE,) \
            .statistic_type(statistic_type) \
            .head_type(head_type) \
            .print_model_characteristics()
        CmdRunner(builder).run(f'non-decomposable-{head_type}-heads_{statistic_type}-statistics')
