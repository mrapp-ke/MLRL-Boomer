"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
# pylint: disable=missing-function-docstring
import pytest

from ..common.cmd_runner import CmdRunner
from ..common.integration_tests import IntegrationTests

from mlrl.common.config.parameters import GlobalPruningParameter, PartitionSamplingParameter

from mlrl.boosting.config.parameters import RegressionLossParameter, StatisticTypeParameter

from mlrl.util.cli import NONE


class BoomerIntegrationTestsMixin(IntegrationTests):
    """
    A mixin for integration tests for the BOOMER algorithm.
    """

    @pytest.mark.parametrize('loss', [
        RegressionLossParameter.LOSS_SQUARED_ERROR_DECOMPOSABLE,
        RegressionLossParameter.LOSS_SQUARED_ERROR_NON_DECOMPOSABLE,
    ])
    @pytest.mark.parametrize('statistic_type', [
        StatisticTypeParameter.STATISTIC_TYPE_FLOAT32,
        StatisticTypeParameter.STATISTIC_TYPE_FLOAT64,
    ])
    def test_loss_squared_error(self, loss: str, statistic_type: str):
        builder = self._create_cmd_builder() \
            .loss(loss) \
            .statistic_type(statistic_type)
        CmdRunner(builder).run(f'loss-{loss}_{statistic_type}-statistics')

    def test_no_default_rule(self):
        builder = self._create_cmd_builder() \
            .default_rule(False) \
            .print_model_characteristics()
        CmdRunner(builder).run('no-default-rule')

    @pytest.mark.parametrize('global_pruning', [
        GlobalPruningParameter.GLOBAL_PRUNING_POST,
        GlobalPruningParameter.GLOBAL_PRUNING_PRE,
    ])
    @pytest.mark.parametrize('holdout', [NONE, PartitionSamplingParameter.PARTITION_SAMPLING_RANDOM])
    def test_global_pruning(self, global_pruning: str, holdout: str):
        builder = self._create_cmd_builder() \
            .global_pruning(global_pruning) \
            .holdout(holdout) \
            .print_model_characteristics()
        CmdRunner(builder).run(f'{global_pruning}_{holdout}-holdout')
