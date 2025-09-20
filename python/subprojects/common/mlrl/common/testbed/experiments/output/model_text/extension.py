"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write rule models to one or several sinks.
"""
from argparse import Namespace
from typing import List, Set, Type, override

from mlrl.common.testbed.experiments.output.model_text.model_text import RuleModelAsText
from mlrl.common.testbed.experiments.output.model_text.writer import RuleModelAsTextWriter

from mlrl.testbed.experiments import Experiment
from mlrl.testbed.experiments.input.sources import TextFileSource
from mlrl.testbed.experiments.output.arguments import OutputArguments, ResultDirectoryArguments
from mlrl.testbed.experiments.output.extension import OutputExtension, ResultDirectoryExtension
from mlrl.testbed.experiments.output.sinks import LogSink, Sink, TextFileSink
from mlrl.testbed.extensions import Extension
from mlrl.testbed.modes import BatchMode, Mode, ReadMode, RunMode, SingleMode

from mlrl.util.cli import Argument, BoolArgument


class RuleModelAsTextExtension(Extension):
    """
    An extension that configures the functionality to write rule models to one or several sinks.
    """

    @staticmethod
    def __create_argument_print_rules(mode: Mode) -> BoolArgument:
        return BoolArgument(
            '--print-rules',
            description='Whether the induced rules should be printed on the console or not.',
            true_options=None if isinstance(mode, ReadMode) else {
                RuleModelAsText.OPTION_PRINT_FEATURE_NAMES, RuleModelAsText.OPTION_PRINT_OUTPUT_NAMES,
                RuleModelAsText.OPTION_PRINT_NOMINAL_VALUES, RuleModelAsText.OPTION_PRINT_BODIES,
                RuleModelAsText.OPTION_PRINT_HEADS, RuleModelAsText.OPTION_DECIMALS_BODY,
                RuleModelAsText.OPTION_DECIMALS_HEAD
            },
        )

    @staticmethod
    def __create_argument_save_rules(mode: Mode) -> BoolArgument:
        return BoolArgument(
            '--save-rules',
            description='Whether the induced rules should be written to a text file or not.',
            true_options=None if isinstance(mode, ReadMode) else {
                RuleModelAsText.OPTION_PRINT_FEATURE_NAMES, RuleModelAsText.OPTION_PRINT_OUTPUT_NAMES,
                RuleModelAsText.OPTION_PRINT_NOMINAL_VALUES, RuleModelAsText.OPTION_PRINT_BODIES,
                RuleModelAsText.OPTION_PRINT_HEADS, RuleModelAsText.OPTION_DECIMALS_BODY,
                RuleModelAsText.OPTION_DECIMALS_HEAD
            },
        )

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(OutputExtension(), ResultDirectoryExtension(), *dependencies)

    @override
    def _get_arguments(self, mode: Mode) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.__create_argument_print_rules(mode), self.__create_argument_save_rules(mode)}

    def __create_log_sinks(self, args: Namespace, mode: Mode) -> List[Sink]:
        value, options = self.__create_argument_print_rules(mode).get_value_and_options(
            args, default=OutputArguments.PRINT_ALL.get_value(args))

        if value:
            return [LogSink(options=options, source_factory=TextFileSource)]
        return []

    def __create_text_file_sinks(self, args: Namespace, mode: Mode) -> List[Sink]:
        value, options = self.__create_argument_save_rules(mode).get_value_and_options(
            args, default=OutputArguments.SAVE_ALL.get_value(args))
        base_dir = OutputArguments.BASE_DIR.get_value(args)
        result_directory = ResultDirectoryArguments.RESULT_DIR.get_value(args)

        if value and base_dir and result_directory:
            return [
                TextFileSink(directory=base_dir / result_directory,
                             create_directory=OutputArguments.CREATE_DIRS.get_value(args),
                             options=options)
            ]
        return []

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder, mode: Mode):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args, mode) + self.__create_text_file_sinks(args, mode)

        if sinks:
            writer = RuleModelAsTextWriter().add_sinks(*sinks)
            experiment_builder.add_post_training_output_writers(writer)

    @override
    def get_supported_modes(self) -> Set[Type[Mode]]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {SingleMode, BatchMode, ReadMode, RunMode}
