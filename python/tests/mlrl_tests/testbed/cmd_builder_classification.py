"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from ..cmd_builder import CmdBuilder
from ..cmd_builder_classification import ClassificationCmdBuilder
from ..datasets import Dataset


class TestbedClassifierCmdBuilder(ClassificationCmdBuilder):
    """
    A builder that allows to configure a command for applying the package mlrl-testbed to classification problems.
    """

    def __init__(self, dataset: str = Dataset.EMOTIONS):
        super().__init__(expected_output_dir=CmdBuilder.EXPECTED_OUTPUT_DIR / 'testbed' / 'classification',
                         input_dir=CmdBuilder.INPUT_DIR / 'testbed',
                         batch_config=CmdBuilder.CONFIG_DIR / 'testbed' / 'classification' / 'batch_config.yml',
                         runnable_module_name='mlrl.boosting',
                         dataset=dataset)
