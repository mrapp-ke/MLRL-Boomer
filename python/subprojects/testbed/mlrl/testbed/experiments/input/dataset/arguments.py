"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines command line arguments for configuring the functionality to load datasets.
"""
from mlrl.util.cli import StringArgument


class DatasetArguments:
    """
    Defines command line arguments for configuring the functionality to load datasets.
    """

    DATASET_NAME = StringArgument(
        '--dataset',
        required=True,
        description='The name of the dataset.',
    )
