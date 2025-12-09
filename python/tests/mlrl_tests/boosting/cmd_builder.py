"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from typing import Optional

from ..common.cmd_builder_rule_learners import RuleLearnerCmdBuilderMixin

from mlrl.common.config.parameters import GlobalPruningParameter

from mlrl.boosting.config.parameters import HeadTypeParameter, StatisticTypeParameter


class BoomerCmdBuilderMixin(RuleLearnerCmdBuilderMixin):
    """
    A mixin for builders that allow to configure a command for running the BOOMER algorithm.
    """

    def loss(self, loss: Optional[str]):
        """
        Configures the algorithm to use a specific loss function.

        :param loss:    The name of the loss function that should be used
        :return:        The builder itself
        """
        if loss:
            self.add_algorithmic_argument('--loss', loss)

        return self

    def default_rule(self, default_rule: bool = True):
        """
        Configures whether the algorithm should induce a default rule or not.

        :param default_rule:    True, if a default rule should be induced, False otherwise
        :return:                The builder itself
        """
        self.add_algorithmic_argument('--default-rule', str(default_rule).lower())
        return self

    def head_type(self, head_type: Optional[str] = HeadTypeParameter.HEAD_TYPE_SINGLE):
        """
        Configures the algorithm to use a specific type of rule heads.

        :param head_type:   The type of rule heads to be used
        :return:            The builder itself
        """
        if head_type:
            self.add_algorithmic_argument('--head-type', head_type)

        return self

    def sparse_statistic_format(self, sparse: bool = True):
        """
        Configures whether sparse data structures should be used to store the statistics or not.

        :param sparse:  True, if sparse data structures should be used to store the statistics, False otherwise
        :return:        The builder itself
        """
        self.add_algorithmic_argument('--statistic-format', 'sparse' if sparse else 'dense')
        return self

    def statistic_type(self, statistic_type: Optional[str] = StatisticTypeParameter.STATISTIC_TYPE_FLOAT64):
        """
        Configures the data type that should be used for representing gradients and Hessians.

        :param statistic_type:  The type that should be used for representing gradients and Hessians
        :return:                The builder itself
        """
        if statistic_type:
            self.add_algorithmic_argument('--statistic-type', statistic_type)

        return self

    def global_pruning(self, global_pruning: Optional[str] = GlobalPruningParameter.GLOBAL_PRUNING_POST):
        """
        Configures the algorithm to use a specific method for pruning entire rules.

        :param global_pruning:  The name of the method that should be used for pruning entire rules
        :return:                The builder itself
        """
        if global_pruning:
            self.add_algorithmic_argument('--global-pruning', global_pruning)

        return self
