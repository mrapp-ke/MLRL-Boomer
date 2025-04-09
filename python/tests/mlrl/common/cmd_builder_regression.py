"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from typing import Optional

from .cmd_builder import DATASET_ATP7D, CmdBuilder


class RegressionCmdBuilder(CmdBuilder):
    """
    A builder that allows to configure a command for applying a rule learning algorithm to a regression problem.
    """

    def __init__(self,
                 expected_output_dir: str,
                 model_file_name: str,
                 runnable_module_name: str,
                 runnable_class_name: Optional[str] = None,
                 dataset: str = DATASET_ATP7D):
        super().__init__(expected_output_dir=expected_output_dir,
                         model_file_name=model_file_name,
                         runnable_module_name=runnable_module_name,
                         runnable_class_name=runnable_class_name,
                         dataset=dataset)
        self.args.append('--problem-type')
        self.args.append('regression')
