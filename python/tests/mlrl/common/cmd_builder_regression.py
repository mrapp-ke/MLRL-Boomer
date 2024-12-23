"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from typing import Optional

from .cmd_builder import DATASET_ATP7D, DIR_DATA, CmdBuilder


class RegressionCmdBuilder(CmdBuilder):
    """
    A builder that allows to configure a command for applying a rule learning algorithm to a regression problem.
    """

    def __init__(self,
                 callback: CmdBuilder.AssertionCallback,
                 expected_output_dir: str,
                 model_file_name: str,
                 runnable_module_name: str,
                 runnable_class_name: Optional[str] = None,
                 data_dir: str = DIR_DATA,
                 dataset: str = DATASET_ATP7D):
        super().__init__(callback=callback,
                         expected_output_dir=expected_output_dir,
                         model_file_name=model_file_name,
                         runnable_module_name=runnable_module_name,
                         runnable_class_name=runnable_class_name,
                         data_dir=data_dir,
                         dataset=dataset)
        self.args.append('--problem-type')
        self.args.append('regression')
