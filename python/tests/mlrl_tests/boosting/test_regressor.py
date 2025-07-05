"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
# pylint: disable=missing-function-docstring
from typing import Any

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

    def _create_cmd_builder(self, dataset: str = Dataset.ATP7D) -> Any:
        return BoomerRegressorCmdBuilder(dataset=dataset)

    def test_decomposable_single_output_heads_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT32) \
            .head_type(HeadTypeParameter.HEAD_TYPE_SINGLE) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-single-output-heads_32-bit-statistics')

    def test_decomposable_single_output_heads_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT64) \
            .head_type(HeadTypeParameter.HEAD_TYPE_SINGLE) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-single-output-heads_64-bit-statistics')

    def test_decomposable_complete_heads_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT32) \
            .head_type(HeadTypeParameter.HEAD_TYPE_COMPLETE) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-complete-heads_32-bit-statistics')

    def test_decomposable_complete_heads_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT64) \
            .head_type(HeadTypeParameter.HEAD_TYPE_COMPLETE) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-complete-heads_64-bit-statistics')

    def test_decomposable_partial_fixed_heads_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT32) \
            .head_type(HeadTypeParameter.HEAD_TYPE_PARTIAL_FIXED) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-partial-fixed-heads_32-bit-statistics')

    def test_decomposable_partial_fixed_heads_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT64) \
            .head_type(HeadTypeParameter.HEAD_TYPE_PARTIAL_FIXED) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-partial-fixed-heads_64-bit-statistics')

    def test_decomposable_partial_dynamic_heads_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT32) \
            .head_type(HeadTypeParameter.HEAD_TYPE_PARTIAL_DYNAMIC) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-partial-dynamic-heads_32-bit-statistics')

    def test_decomposable_partial_dynamic_heads_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT64) \
            .head_type(HeadTypeParameter.HEAD_TYPE_PARTIAL_DYNAMIC) \
            .print_model_characteristics()
        CmdRunner(builder).run('decomposable-partial-dynamic-heads_64-bit-statistics')

    def test_non_decomposable_single_label_heads_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_NON_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT32) \
            .head_type(HeadTypeParameter.HEAD_TYPE_SINGLE) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-single-output-heads_32-bit-statistics')

    def test_non_decomposable_single_label_heads_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_NON_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT64) \
            .head_type(HeadTypeParameter.HEAD_TYPE_SINGLE) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-single-output-heads_64-bit-statistics')

    def test_non_decomposable_complete_heads_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_NON_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT32) \
            .head_type(HeadTypeParameter.HEAD_TYPE_COMPLETE) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-complete-heads_32-bit-statistics')

    def test_non_decomposable_complete_heads_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_NON_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT64) \
            .head_type(HeadTypeParameter.HEAD_TYPE_COMPLETE) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-complete-heads_64-bit-statistics')

    def test_non_decomposable_partial_fixed_heads_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_NON_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT32) \
            .head_type(HeadTypeParameter.HEAD_TYPE_PARTIAL_FIXED) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-partial-fixed-heads_32-bit-statistics')

    def test_non_decomposable_partial_fixed_heads_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_NON_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT64) \
            .head_type(HeadTypeParameter.HEAD_TYPE_PARTIAL_FIXED) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-partial-fixed-heads_64-bit-statistics')

    def test_non_decomposable_partial_dynamic_heads_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_NON_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT32) \
            .head_type(HeadTypeParameter.HEAD_TYPE_PARTIAL_DYNAMIC) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-partial-dynamic-heads_32-bit-statistics')

    def test_non_decomposable_partial_dynamic_heads_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_NON_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT64) \
            .head_type(HeadTypeParameter.HEAD_TYPE_PARTIAL_DYNAMIC) \
            .print_model_characteristics()
        CmdRunner(builder).run('non-decomposable-partial-dynamic-heads_64-bit-statistics')
