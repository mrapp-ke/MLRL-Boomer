"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from ..common.cmd_builder import CmdBuilder
from ..common.cmd_runner import CmdRunner
from .cmd_builder import BoomerCmdBuilderMixin


class BoomerIntegrationTestsMixin:
    """
    A mixin for integration tests for the BOOMER algorithm.
    """

    def test_loss_squared_error_decomposable_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using the decomposable squared error loss function and 32-bit statistics.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_SQUARED_ERROR_DECOMPOSABLE) \
            .statistic_type(STATISTIC_TYPE_FLOAT32)
        CmdRunner(self, builder).run('loss-squared-error-decomposable_32-bit-statistics')

    def test_loss_squared_error_decomposable_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using the decomposable squared error loss function and 64-bit statistics.
        """
        builder = self._create_cmd_builder() \
            .loss(BoomerCmdBuilderMixin.LOSS_SQUARED_ERROR_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64)
        CmdRunner(self, builder).run('loss-squared-error-decomposable_64-bit-statistics')

    def test_loss_squared_error_non_decomposable_32bit_statistics(self):
        """
        Tests the BOOMER algorithm when using the non-decomposable squared error loss function and 32-bit statistics.
        """
        builder = self._create_cmd_builder() \
            .loss(BoomerCmdBuilderMixin.LOSS_SQUARED_ERROR_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT32)
        CmdRunner(self, builder).run('loss-squared-error-non-decomposable_32-bit-statistics')

    def test_loss_squared_error_non_decomposable_64bit_statistics(self):
        """
        Tests the BOOMER algorithm when using the non-decomposable squared error loss function and 64-bit statistics.
        """
        builder = self._create_cmd_builder() \
            .loss(BoomerCmdBuilderMixin.LOSS_SQUARED_ERROR_NON_DECOMPOSABLE) \
            .statistic_type(BoomerCmdBuilderMixin.STATISTIC_TYPE_FLOAT64)
        CmdRunner(self, builder).run('loss-squared-error-non-decomposable_64-bit-statistics')

    def test_no_default_rule(self):
        """
        Tests the BOOMER algorithm when not inducing a default rule.
        """
        builder = self._create_cmd_builder() \
            .default_rule(False) \
            .print_model_characteristics()
        CmdRunner(self, builder).run('no-default-rule')

    def test_global_post_pruning_no_holdout(self):
        """
        Tests the BOOMER algorithm when using no holdout set for global post-pruning.
        """
        builder = self._create_cmd_builder() \
            .global_pruning(BoomerCmdBuilderMixin.GLOBAL_PRUNING_POST) \
            .holdout(CmdBuilder.HOLDOUT_NO) \
            .print_model_characteristics()
        CmdRunner(self, builder).run('post-pruning_no-holdout')

    def test_global_post_pruning_random_holdout(self):
        """
        Tests the BOOMER algorithm when using a holdout set that is created via random sampling for global post-pruning.
        """
        builder = self._create_cmd_builder() \
            .global_pruning(BoomerCmdBuilderMixin.GLOBAL_PRUNING_POST) \
            .holdout(CmdBuilder.HOLDOUT_RANDOM) \
            .print_model_characteristics()
        CmdRunner(self, builder).run('post-pruning_random-holdout')

    def test_global_pre_pruning_no_holdout(self):
        """
        Tests the BOOMER algorithm when using no holdout set for global pre-pruning.
        """
        builder = self._create_cmd_builder() \
            .global_pruning(BoomerCmdBuilderMixin.GLOBAL_PRUNING_PRE) \
            .holdout(CmdBuilder.HOLDOUT_NO) \
            .print_model_characteristics()
        CmdRunner(self, builder).run('pre-pruning_no-holdout')

    def test_global_pre_pruning_random_holdout(self):
        """
        Tests the BOOMER algorithm when using a holdout set that is created via random sampling for global pre-pruning.
        """
        builder = self._create_cmd_builder() \
            .global_pruning(BoomerCmdBuilderMixin.GLOBAL_PRUNING_PRE) \
            .holdout(CmdBuilder.HOLDOUT_RANDOM) \
            .print_model_characteristics()
        CmdRunner(self, builder).run('pre-pruning_random-holdout')
