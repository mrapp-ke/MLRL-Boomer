"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from ..common.cmd_builder import CmdBuilder
from ..common.cmd_builder_regression import RegressionCmdBuilder
from ..common.datasets import Dataset
from .cmd_builder import BoomerCmdBuilderMixin


class BoomerRegressorCmdBuilder(RegressionCmdBuilder, BoomerCmdBuilderMixin):
    """
    A builder that allows to configure a command for running the BOOMER algorithm for regression problems.
    """

    def __init__(self, dataset: str = Dataset.ATP7D):
        super().__init__(expected_output_dir=CmdBuilder.EXPECTED_OUTPUT_DIR / 'boosting' / 'regression',
                         input_dir=CmdBuilder.INPUT_DIR / 'boosting',
                         batch_config=CmdBuilder.CONFIG_DIR / 'boosting' / 'regression' / 'batch_config.yml',
                         runnable_module_name='mlrl.boosting',
                         dataset=dataset)
