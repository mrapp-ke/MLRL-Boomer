"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write output data to one or several sinks.
"""
from argparse import Namespace
from os import listdir, path, unlink
from typing import List

from mlrl.testbed.cli import Argument
from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.extensions.extension import Extension


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

    PARAM_OUTPUT_DIR = '--output-dir'

    def get_arguments(self) -> List[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_arguments`
        """
        return [
            Argument.string(
                self.PARAM_OUTPUT_DIR,
                help='The path to the directory where experimental results should be saved.',
            ),
            Argument.bool('--wipe-output-dir',
                          default=True,
                          help='Whether all files in the directory specified via the argument '
                          + OutputExtension.PARAM_OUTPUT_DIR
                          + ' should be deleted before an experiment starts or not.'),
            Argument.bool(
                '--exit-on-error',
                default=False,
                help='Whether the program should exit if an error occurs while writing experimental results or not.',
            ),
        ]

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        experiment_builder.set_exit_on_error(args.exit_on_error)
        output_dir = args.output_dir

        if output_dir and args.wipe_output_dir:
            listener = OutputExtension.WipeDirectoryListener(output_dir)
            experiment_builder.add_listeners(listener)
