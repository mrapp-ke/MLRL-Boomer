"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from ..common.cmd_builder import HOLDOUT_NO, HOLDOUT_RANDOM
from ..common.decorators import skip_test_on_ci
from .cmd_builder import GLOBAL_PRUNING_POST, GLOBAL_PRUNING_PRE, LOSS_SQUARED_ERROR_DECOMPOSABLE, \
    LOSS_SQUARED_ERROR_NON_DECOMPOSABLE


class BoomerIntegrationTestsMixin:
    """
    A mixin for integration tests for the BOOMER algorithm.
    """

    def test_loss_squared_error_decomposable(self):
        """
        Tests the BOOMER algorithm when using the decomposable squared error loss function.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_SQUARED_ERROR_DECOMPOSABLE)
        builder.run_cmd('loss-squared-error-decomposable')

    @skip_test_on_ci
    def test_loss_squared_error_non_decomposable(self):
        """
        Tests the BOOMER algorithm when using the non-decomposable squared error loss function.
        """
        builder = self._create_cmd_builder() \
            .loss(LOSS_SQUARED_ERROR_NON_DECOMPOSABLE)
        builder.run_cmd('loss-squared-error-non-decomposable')

    def test_no_default_rule(self):
        """
        Tests the BOOMER algorithm when not inducing a default rule.
        """
        builder = self._create_cmd_builder() \
            .default_rule(False) \
            .print_model_characteristics()
        builder.run_cmd('no-default-rule')

    def test_global_post_pruning_no_holdout(self):
        """
        Tests the BOOMER algorithm when using no holdout set for global post-pruning.
        """
        builder = self._create_cmd_builder() \
            .global_pruning(GLOBAL_PRUNING_POST) \
            .holdout(HOLDOUT_NO) \
            .print_model_characteristics()
        builder.run_cmd('post-pruning_no-holdout')

    def test_global_post_pruning_random_holdout(self):
        """
        Tests the BOOMER algorithm when using a holdout set that is created via random sampling for global post-pruning.
        """
        builder = self._create_cmd_builder() \
            .global_pruning(GLOBAL_PRUNING_POST) \
            .holdout(HOLDOUT_RANDOM) \
            .print_model_characteristics()
        builder.run_cmd('post-pruning_random-holdout')

    def test_global_pre_pruning_no_holdout(self):
        """
        Tests the BOOMER algorithm when using no holdout set for global pre-pruning.
        """
        builder = self._create_cmd_builder() \
            .global_pruning(GLOBAL_PRUNING_PRE) \
            .holdout(HOLDOUT_NO) \
            .print_model_characteristics()
        builder.run_cmd('pre-pruning_no-holdout')

    def test_global_pre_pruning_random_holdout(self):
        """
        Tests the BOOMER algorithm when using a holdout set that is created via random sampling for global pre-pruning.
        """
        builder = self._create_cmd_builder() \
            .global_pruning(GLOBAL_PRUNING_PRE) \
            .holdout(HOLDOUT_RANDOM) \
            .print_model_characteristics()
        builder.run_cmd('pre-pruning_random-holdout')