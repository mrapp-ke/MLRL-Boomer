"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write ground truth to one or several sinks.
"""
from argparse import Namespace
from pathlib import Path
from typing import Set, override

from mlrl.testbed_arff.experiments.output.sinks.sink_arff import ArffFileSink

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.arguments import OutputArguments, ResultDirectoryArguments
from mlrl.testbed.experiments.output.extension import OutputExtension, ResultDirectoryExtension
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.util.format import OPTION_DECIMALS

from mlrl.util.cli import Argument, BoolArgument


class GroundTruthExtension(Extension):
    """
    An extension that configures the functionality to write ground truth to one or several sinks.
    """

    PRINT_GROUND_TRUTH = BoolArgument(
        '--print-ground-truth',
        description='Whether the ground truth should be printed on the console or not.',
        true_options={OPTION_DECIMALS},
    )

    SAVE_GROUND_TRUTH = BoolArgument(
        '--save-ground-truth',
        description='Whether the ground truth should be written to output files or not.',
        true_options={OPTION_DECIMALS},
    )

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(OutputExtension(), ResultDirectoryExtension(), *dependencies)

    @override
    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.PRINT_GROUND_TRUTH, self.SAVE_GROUND_TRUTH}

    def __configure_log_sink(self, args: Namespace, experiment_builder: Experiment.Builder):
        print_all = OutputArguments.PRINT_ALL.get_value(args)
        print_ground_truth, options = self.PRINT_GROUND_TRUTH.get_value(args, default=print_all)

        if print_ground_truth:
            experiment_builder.ground_truth_writer.add_sinks(LogSink(options=options))

    def __configure_arff_file_sink(self, args: Namespace, experiment_builder: Experiment.Builder):
        save_all = OutputArguments.SAVE_ALL.get_value(args)
        save_ground_truth, options = self.SAVE_GROUND_TRUTH.get_value(args, default=save_all)
        result_directory = ResultDirectoryArguments.RESULT_DIR.get_value(args)

        if save_ground_truth and result_directory:
            create_directory = OutputArguments.CREATE_DIRS.get_value(args)
            experiment_builder.ground_truth_writer.add_sinks(
                ArffFileSink(directory=Path(result_directory), create_directory=create_directory, options=options))

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        self.__configure_log_sink(args, experiment_builder)
        self.__configure_arff_file_sink(args, experiment_builder)
