"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from os import path

from ..common.cmd_builder import DATASET_ATP7D, DIR_OUT, CmdBuilder
from ..common.cmd_builder_regression import RegressionCmdBuilder
from .cmd_builder import BoomerCmdBuilderMixin


class BoomerRegressorCmdBuilder(RegressionCmdBuilder, BoomerCmdBuilderMixin):
    """
    A builder that allows to configure a command for running the BOOMER algorithm for regression problems.
    """

    def __init__(self, callback: CmdBuilder.AssertionCallback, dataset: str = DATASET_ATP7D):
        super().__init__(callback,
                         expected_output_dir=path.join(DIR_OUT, 'boomer-regressor'),
                         model_file_name='boomer',
                         runnable_module_name='mlrl.boosting',
                         dataset=dataset)
