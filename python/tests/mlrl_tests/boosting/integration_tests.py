"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
# pylint: disable=missing-function-docstring
from ..common.cmd_runner import CmdRunner

from mlrl.common.config.parameters import GlobalPruningParameter, PartitionSamplingParameter

from mlrl.boosting.config.parameters import RegressionLossParameter, StatisticTypeParameter

from mlrl.util.cli import NONE


class BoomerIntegrationTestsMixin:
    """
    A mixin for integration tests for the BOOMER algorithm.
    """

    def test_loss_squared_error_decomposable_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT32)
        CmdRunner(builder).run('loss-squared-error-decomposable_32-bit-statistics')

    def test_loss_squared_error_decomposable_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT64)
        CmdRunner(builder).run('loss-squared-error-decomposable_64-bit-statistics')

    def test_loss_squared_error_non_decomposable_32bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_NON_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT32)
        CmdRunner(builder).run('loss-squared-error-non-decomposable_32-bit-statistics')

    def test_loss_squared_error_non_decomposable_64bit_statistics(self):
        builder = self._create_cmd_builder() \
            .loss(RegressionLossParameter.LOSS_SQUARED_ERROR_NON_DECOMPOSABLE) \
            .statistic_type(StatisticTypeParameter.STATISTIC_TYPE_FLOAT64)
        CmdRunner(builder).run('loss-squared-error-non-decomposable_64-bit-statistics')

    def test_no_default_rule(self):
        builder = self._create_cmd_builder() \
            .default_rule(False) \
            .print_model_characteristics()
        CmdRunner(builder).run('no-default-rule')

    def test_global_post_pruning_no_holdout(self):
        builder = self._create_cmd_builder() \
            .global_pruning(GlobalPruningParameter.GLOBAL_PRUNING_POST) \
            .holdout(NONE) \
            .print_model_characteristics()
        CmdRunner(builder).run('post-pruning_no-holdout')

    def test_global_post_pruning_random_holdout(self):
        builder = self._create_cmd_builder() \
            .global_pruning(GlobalPruningParameter.GLOBAL_PRUNING_POST) \
            .holdout(PartitionSamplingParameter.PARTITION_SAMPLING_RANDOM) \
            .print_model_characteristics()
        CmdRunner(builder).run('post-pruning_random-holdout')

    def test_global_pre_pruning_no_holdout(self):
        builder = self._create_cmd_builder() \
            .global_pruning(GlobalPruningParameter.GLOBAL_PRUNING_PRE) \
            .holdout(NONE) \
            .print_model_characteristics()
        CmdRunner(builder).run('pre-pruning_no-holdout')

    def test_global_pre_pruning_random_holdout(self):
        builder = self._create_cmd_builder() \
            .global_pruning(GlobalPruningParameter.GLOBAL_PRUNING_PRE) \
            .holdout(PartitionSamplingParameter.PARTITION_SAMPLING_RANDOM) \
            .print_model_characteristics()
        CmdRunner(builder).run('pre-pruning_random-holdout')
