"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write output data to one or several sinks.
"""
from argparse import ArgumentParser, Namespace
from os import listdir, path, unlink

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.format import format_iterable
from mlrl.util.options import BooleanOption


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

    def configure_arguments(self, argument_parser: ArgumentParser):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_arguments`
        """
        argument_parser.add_argument(self.PARAM_OUTPUT_DIR,
                                     type=str,
                                     help='The path to the directory where experimental results should be saved.')
        argument_parser.add_argument('--wipe-output-dir',
                                     type=BooleanOption.parse,
                                     default=True,
                                     help='Whether all files in the directory specified via the argument '
                                     + OutputExtension.PARAM_OUTPUT_DIR + ' should be deleted before an experiment '
                                     + 'starts or not. Must be one of ' + format_iterable(BooleanOption) + '.')

    def configure_experiment(self, args: Namespace, experiment: Experiment):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        output_dir = args.output_dir

        if output_dir and args.wipe_output_dir:
            listener = OutputExtension.WipeDirectoryListener(output_dir)
            experiment.add_listeners(listener)
