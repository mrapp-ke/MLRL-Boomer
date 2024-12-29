"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from os import path

from ..common.cmd_builder import DATASET_EMOTIONS, DIR_OUT, CmdBuilder
from ..common.cmd_builder_classification import ClassificationCmdBuilder

HEURISTIC_ACCURACY = 'accuracy'

HEURISTIC_PRECISION = 'precision'

HEURISTIC_LAPLACE = 'laplace'

HEURISTIC_RECALL = 'recall'

HEURISTIC_WRA = 'weighted-relative-accuracy'

HEURISTIC_F_MEASURE = 'f-measure'

HEURISTIC_M_ESTIMATE = 'm-estimate'

HEAD_TYPE_SINGLE = 'single'

HEAD_TYPE_PARTIAL = 'partial'

LIFT_FUNCTION_NO = 'none'

LIFT_FUNCTION_PEAK = 'peak'

LIFT_FUNCTION_KLN = 'kln'


class SeCoClassifierCmdBuilder(ClassificationCmdBuilder):
    """
    A builder that allows to configure a command for running the separate-and-conquer (SeCo) algorithm.
    """

    def __init__(self, callback: CmdBuilder.AssertionCallback, dataset: str = DATASET_EMOTIONS):
        super().__init__(callback,
                         expected_output_dir=path.join(DIR_OUT, 'seco', 'classification'),
                         model_file_name='seco',
                         runnable_module_name='mlrl.seco',
                         dataset=dataset)

    def heuristic(self, heuristic: str = HEURISTIC_F_MEASURE):
        """
        Configures the algorithm to use a specific heuristic for learning rules.

        :param heuristic:   The name of the heuristic that should be used for learning rules
        :return:            The builder itself
        """
        self.args.append('--heuristic')
        self.args.append(heuristic)
        return self

    def pruning_heuristic(self, heuristic: str = HEURISTIC_ACCURACY):
        """
        Configures the algorithm to use a specific heuristic for pruning rules.

        :param heuristic:   The name of the heuristic that should be used for pruning rules
        :return:            The builder itself
        """
        self.args.append('--pruning-heuristic')
        self.args.append(heuristic)
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

    def lift_function(self, lift_function: str = LIFT_FUNCTION_PEAK):
        """
        Configures the algorithm to use a specific lift function for the induction of rules with partial heads.

        :param lift_function:   The name of the lift function that should be used for the induction of rules with
                                partial heads
        :return:                The builder itself
        """
        self.args.append('--lift-function')
        self.args.append(lift_function)
        return self
