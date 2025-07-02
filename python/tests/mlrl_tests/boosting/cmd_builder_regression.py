"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from os import path

from ..common.cmd_builder_regression import RegressionCmdBuilder
from ..common.datasets import Dataset
from .cmd_builder import BoomerCmdBuilderMixin


class BoomerRegressorCmdBuilder(RegressionCmdBuilder, BoomerCmdBuilderMixin):
    """
    A builder that allows to configure a command for running the BOOMER algorithm for regression problems.
    """

    def __init__(self, dataset: str = Dataset.ATP7D):
        super().__init__(expected_output_dir=path.join('boosting', 'regression'),
                         runnable_module_name='mlrl.boosting',
                         dataset=dataset)
