"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from ..cmd_builder import CmdBuilder
from ..cmd_builder_regression import RegressionCmdBuilder
from ..datasets import Dataset


class RegressionTestbedCmdBuilder(RegressionCmdBuilder):
    """
    A builder that allows to configure a command for applying the package mlrl-testbed to regression problems.
    """

    def __init__(self, dataset: str = Dataset.ATP7D):
        super().__init__(expected_output_dir=CmdBuilder.EXPECTED_OUTPUT_DIR / 'testbed' / 'regression',
                         input_dir=CmdBuilder.INPUT_DIR / 'testbed',
                         batch_config=CmdBuilder.CONFIG_DIR / 'testbed' / 'regression' / 'batch_config.yml',
                         runnable_module_name='mlrl.boosting',
                         dataset=dataset)
