"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write output data to one or several sinks.
"""
from argparse import Namespace
from os import listdir, path, unlink
from typing import Set

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, BoolArgument, StringArgument


class OutputExtension(Extension):
    """
    An extension that configures the functionality to write output data to one or several sinks.
    """

    class WipeDirectoryListener(Experiment.Listener):
        """
        Deletes all files from a directory before an experiment starts.
        """

        def __init__(self, directory: str):
            """
            :param directory: The path to the directory from which the files should be deleted
            """
            self.directory = directory

        def before_start(self, _: Experiment):
            """
            See :func:`mlrl.testbed.experiments.Experiment.Listener.before_start`
            """
            directory = self.directory

            if path.isdir(directory):
                for file in listdir(directory):
                    file_path = path.join(directory, file)

                    if path.isfile(file_path):
                        unlink(file_path)

    OUTPUT_DIR = StringArgument(
        '--output-dir',
        description='The path to the directory where experimental results should be saved.',
    )

    CREATE_OUTPUT_DIR = BoolArgument(
        '--create-output-dir',
        default=True,
        description='Whether the directory specified via the argument ' + OUTPUT_DIR.name + ' should automatically be '
        + 'created, if it does not exist, or not.',
    )

    WIPE_OUTPUT_DIR = BoolArgument(
        '--wipe-output-dir',
        default=True,
        description='Whether all files in the directory specified via the argument ' + OUTPUT_DIR.name + ' should be '
        + 'deleted before an experiment starts or not.',
    )

    EXIT_ON_ERROR = BoolArgument(
        '--exit-on-error',
        default=False,
        description='Whether the program should exit if an error occurs while writing experimental results or not.',
    )

    PRINT_ALL = BoolArgument(
        '--print-all',
        default=False,
        description='Whether all output data should be printed on the console or not.',
    )

    STORE_ALL = BoolArgument(
        '--store-all',
        default=False,
        description='Whether all output data should be written to files or not.',
    )

    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {
            self.OUTPUT_DIR, self.CREATE_OUTPUT_DIR, self.WIPE_OUTPUT_DIR, self.EXIT_ON_ERROR, self.PRINT_ALL,
            self.STORE_ALL
        }

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        experiment_builder.set_exit_on_error(self.EXIT_ON_ERROR.get_value(args))
        output_dir = self.OUTPUT_DIR.get_value(args)

        if output_dir and self.WIPE_OUTPUT_DIR.get_value(args):
            listener = OutputExtension.WipeDirectoryListener(output_dir)
            experiment_builder.add_listeners(listener)
