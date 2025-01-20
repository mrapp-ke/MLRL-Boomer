"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""

LOSS_SQUARED_ERROR_DECOMPOSABLE = 'squared-error-decomposable'

LOSS_SQUARED_ERROR_NON_DECOMPOSABLE = 'squared-error-non-decomposable'

HEAD_TYPE_SINGLE = 'single'

HEAD_TYPE_COMPLETE = 'complete'

HEAD_TYPE_PARTIAL_FIXED = 'partial-fixed'

HEAD_TYPE_PARTIAL_DYNAMIC = 'partial-dynamic'

GLOBAL_PRUNING_PRE = 'pre-pruning'

GLOBAL_PRUNING_POST = 'post-pruning'

STATISTIC_TYPE_FLOAT32 = 'float32'

STATISTIC_TYPE_FLOAT64 = 'float64'


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

    def statistic_type(self, statistic_type: str = STATISTIC_TYPE_FLOAT64):
        """
        Configures the data type that should be used for representing gradients and Hessians.

        :param statistic_type:  The type that should be used for representing gradients and Hessians
        :return:                The builder itself
        """
        self.args.append('--statistic-type')
        self.args.append(statistic_type)
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
