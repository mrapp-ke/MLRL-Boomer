"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines command line arguments for configuring the functionality to run experiments via the Slurm Workload Manager.
"""
from mlrl.util.cli import BoolArgument, StringArgument


class SlurmArguments:
    """
    Defines command line arguments for configuring the functionality to run experiments via the Slurm Workload Manager.
    """

    SAVE_SLURM_SCRIPTS = BoolArgument(
        '--save-slurm-scripts',
        default=False,
        description='Whether the Slurm scripts for running individual experiments in a batch should be saved to the '
        + 'working directory or not.',
    )

    SLURM_SAVE_DIR = StringArgument(
        '--slurm-save-dir',
        default='.',
        description='An absolute or relative path to the directory where Slurm scripts should be saved.')

    PRINT_SLURM_SCRIPTS = BoolArgument(
        '--print-slurm-scripts',
        default=False,
        description='Whether the Slurm scripts for running individual experiments in a batch should be printed or not.',
    )

    SLURM_CONFIG_FILE = StringArgument(
        '--slurm-config',
        description='An absolute or relative path to a YAML file that configures the Slurm jobs to be run.',
    )
