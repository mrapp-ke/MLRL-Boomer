"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines command line arguments for configuring the directory to which models should be written.
"""
from mlrl.util.cli import PathArgument


class ModelOutputDirectoryArguments:
    """
    Defines command line arguments for configuring the directory to which models should be written.
    """

    MODEL_SAVE_DIR = PathArgument(
        '--model-save-dir',
        default='models',
        description='The path to the directory where models should be saved.',
    )
