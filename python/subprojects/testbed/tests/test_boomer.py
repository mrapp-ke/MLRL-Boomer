"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from test_common import HOLDOUT_NO, HOLDOUT_RANDOM, skip_test_on_ci

LOSS_SQUARED_ERROR_DECOMPOSABLE = 'squared-error-decomposable'

LOSS_SQUARED_ERROR_NON_DECOMPOSABLE = 'squared-error-non-decomposable'

HEAD_TYPE_SINGLE = 'single'

HEAD_TYPE_COMPLETE = 'complete'

HEAD_TYPE_PARTIAL_FIXED = 'partial-fixed'

HEAD_TYPE_PARTIAL_DYNAMIC = 'partial-dynamic'

GLOBAL_PRUNING_PRE = 'pre-pruning'

GLOBAL_PRUNING_POST = 'post-pruning'


class BoomerCmdBuilderMixin:
    """
    A mixin for builders that allow to configure a command for running the BOOMER algorithm.
    """

    def loss(self, loss: str):
        """
        Configures the algorithm to use a specific loss function.

        :param loss:    The name of the loss function that should be used
        :return:        The builder itself
        """
        self.args.append('--loss')
        self.args.append(loss)
        return self

    def default_rule(self, default_rule: bool = True):
        """
        Configures whether the algorithm should induce a default rule or not.

        :param default_rule:    True, if a default rule should be induced, False otherwise
        :return:                The builder itself
        """
        self.args.append('--default-rule')
        self.args.append(str(default_rule).lower())
        return self

    def head_type(self, head_type: str = HEAD_TYPE_SINGLE):
        """
        Configures the algorithm to use a specific type of rule heads.

        :param head_type:   The type of rule heads to be used
        :return:            The builder itself
        """
        self.args.append('--head-type')
        self.args.append(head_type)
        return self

    def sparse_statistic_format(self, sparse: bool = True):
        """
        Configures whether sparse data structures should be used to store the statistics or not.

        :param sparse:  True, if sparse data structures should be used to store the statistics, False otherwise
        :return:        The builder itself
        """
        self.args.append('--statistic-format')
        self.args.append('sparse' if sparse else 'dense')
        return self

    def global_pruning(self, global_pruning: str = GLOBAL_PRUNING_POST):
        """
        Configures the algorithm to use a specific method for pruning entire rules.

        :param global_pruning:  The name of the method that should be used for pruning entire rules
        :return:                The builder itself
        """
        self.args.append('--global-pruning')
        self.args.append(global_pruning)
        return self


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
