"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write evaluation results to one or several sinks.
"""
from argparse import Namespace
from typing import Set, override

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.arguments import OutputArguments
from mlrl.testbed.experiments.output.evaluation.evaluation_result import AggregatedEvaluationResult
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.sinks import CsvFileSink, LogSink
from mlrl.testbed.experiments.state import ExperimentMode
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE

from mlrl.util.cli import Argument, BoolArgument


class AggregatedEvaluationExtension(Extension):
    """
    An extension that configures the functionality to write evaluation results that have been aggregated across several
    experiments to one or several sinks.
    """

    PRINT_EVALUATION = BoolArgument(
        '--print-evaluation',
        default=True,
        description='Whether the evaluation results should be printed on the console or not.',
        true_options={
            AggregatedEvaluationResult.OPTION_ENABLE_ALL, OPTION_DECIMALS, OPTION_PERCENTAGE,
            AggregatedEvaluationResult.OPTION_RANK
        },
    )

    SAVE_EVALUATION = BoolArgument(
        '--save-evaluation',
        description='Whether evaluation results should be written to output files or not.',
        true_options={
            AggregatedEvaluationResult.OPTION_ENABLE_ALL, OPTION_DECIMALS, AggregatedEvaluationResult.OPTION_RANK
        },
    )

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(OutputExtension(), *dependencies)

    @override
    def _get_arguments(self, _: ExperimentMode) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.PRINT_EVALUATION, self.SAVE_EVALUATION}

    def __configure_log_sink(self, args: Namespace, experiment_builder: Experiment.Builder):
        print_all = OutputArguments.PRINT_ALL.get_value(args)
        print_evaluation, options = self.PRINT_EVALUATION.get_value_and_options(args, default=print_all)

        if print_evaluation:
            experiment_builder.aggregated_evaluation_writer.add_sinks(LogSink(options=options))

    def __configure_csv_file_sink(self, args: Namespace, experiment_builder: Experiment.Builder):
        save_all = OutputArguments.SAVE_ALL.get_value(args)
        save_evaluation, options = self.SAVE_EVALUATION.get_value_and_options(args, default=save_all)
        base_dir = OutputArguments.BASE_DIR.get_value(args)

        if base_dir and save_evaluation:
            create_directory = OutputArguments.CREATE_DIRS.get_value(args)
            experiment_builder.aggregated_evaluation_writer.add_sinks(
                CsvFileSink(directory=base_dir, create_directory=create_directory, options=options))

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder, mode: ExperimentMode):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        self.__configure_log_sink(args, experiment_builder)
        self.__configure_csv_file_sink(args, experiment_builder)

    @override
    def get_supported_modes(self) -> Set[ExperimentMode]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {ExperimentMode.READ}
