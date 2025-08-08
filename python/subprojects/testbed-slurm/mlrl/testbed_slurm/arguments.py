"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines command line arguments for configuring the functionality to run experiments via the Slurm Workload Manager.
"""
from mlrl.util.cli import BoolArgument


class SlurmArguments:
    """
    Defines command line arguments for configuring the functionality to run experiments via the Slurm Workload Manager.
    """

    SAVE_SLURM_SCRIPTS = BoolArgument(
        '--save-slurm-scripts',
        default=False,
        description='Whether the SLURM scripts for running individual experiments in a batch should be saved to the '
        + 'working directory or not.')

    PRINT_SLURM_SCRIPTS = BoolArgument(
        '--print-slurm-scripts',
        default=False,
        description='Whether the SLURM scripts for running individual experiments in a batch should be printed or not.')
