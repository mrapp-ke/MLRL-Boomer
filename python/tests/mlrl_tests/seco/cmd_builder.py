"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from os import path
from typing import Optional

from ..common.cmd_builder_classification import ClassificationCmdBuilder
from ..common.datasets import Dataset

from mlrl.seco.config.parameters import HEURISTIC_ACCURACY, HEURISTIC_F_MEASURE, HeadTypeParameter, \
    LiftFunctionParameter


class SeCoClassifierCmdBuilder(ClassificationCmdBuilder):
    """
    A builder that allows to configure a command for running the separate-and-conquer (SeCo) algorithm.
    """

    def __init__(self, dataset: str = Dataset.EMOTIONS):
        super().__init__(expected_output_dir=path.join('seco', 'classification'),
                         runnable_module_name='mlrl.seco',
                         dataset=dataset)

    def heuristic(self, heuristic: Optional[str] = HEURISTIC_F_MEASURE):
        """
        Configures the algorithm to use a specific heuristic for learning rules.

        :param heuristic:   The name of the heuristic that should be used for learning rules
        :return:            The builder itself
        """
        if heuristic:
            self.args.append('--heuristic')
            self.args.append(heuristic)

        return self

    def pruning_heuristic(self, heuristic: Optional[str] = HEURISTIC_ACCURACY):
        """
        Configures the algorithm to use a specific heuristic for pruning rules.

        :param heuristic:   The name of the heuristic that should be used for pruning rules
        :return:            The builder itself
        """
        if heuristic:
            self.args.append('--pruning-heuristic')
            self.args.append(heuristic)

        return self

    def head_type(self, head_type: Optional[str] = HeadTypeParameter.HEAD_TYPE_SINGLE):
        """
        Configures the algorithm to use a specific type of rule heads.

        :param head_type:   The type of rule heads to be used
        :return:            The builder itself
        """
        if head_type:
            self.args.append('--head-type')
            self.args.append(head_type)

        return self

    def lift_function(self, lift_function: Optional[str] = LiftFunctionParameter.LIFT_FUNCTION_PEAK):
        """
        Configures the algorithm to use a specific lift function for the induction of rules with partial heads.

        :param lift_function:   The name of the lift function that should be used for the induction of rules with
                                partial heads
        :return:                The builder itself
        """
        if lift_function:
            self.args.append('--lift-function')
            self.args.append(lift_function)

        return self
