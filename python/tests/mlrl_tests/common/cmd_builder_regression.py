"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from pathlib import Path
from typing import Optional

from .cmd_builder import CmdBuilder
from .datasets import Dataset


class RegressionCmdBuilder(CmdBuilder):
    """
    A builder that allows to configure a command for applying a rule learning algorithm to a regression problem.
    """

    def __init__(self,
                 expected_output_dir: Path,
                 input_dir: Path,
                 batch_config: Path,
                 runnable_module_name: str,
                 runnable_class_name: Optional[str] = None,
                 dataset: str = Dataset.ATP7D):
        super().__init__(expected_output_dir=expected_output_dir,
                         input_dir=input_dir,
                         batch_config=batch_config,
                         runnable_module_name=runnable_module_name,
                         runnable_class_name=runnable_class_name,
                         dataset=dataset)
        self.problem_type = 'regression'
