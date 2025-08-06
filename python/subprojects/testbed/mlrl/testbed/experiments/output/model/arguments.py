"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines command line arguments for configuring the directory to which models should be written.
"""
from pathlib import Path

from mlrl.testbed.experiments.output.arguments import OutputArguments

from mlrl.util.cli import StringArgument


class ModelOutputDirectoryArguments:
    """
    Defines command line arguments for configuring the directory to which models should be written.
    """

    MODEL_SAVE_DIR = StringArgument(
        '--model-save-dir',
        default='models',
        description='The path to the directory where models should be saved.',
        decorator=lambda args, value: Path(OutputArguments.BASE_DIR.get_value(args)) / value,
    )
