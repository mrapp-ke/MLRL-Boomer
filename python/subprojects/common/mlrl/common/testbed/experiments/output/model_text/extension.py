"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write rule models to one or several sinks.
"""
from argparse import Namespace
from pathlib import Path
from typing import List, Set, override

from mlrl.common.testbed.experiments.output.model_text.model_text import RuleModelAsText
from mlrl.common.testbed.experiments.output.model_text.writer import RuleModelAsTextWriter

from mlrl.testbed.experiments import Experiment
from mlrl.testbed.experiments.output.arguments import OutputArguments, ResultDirectoryArguments
from mlrl.testbed.experiments.output.extension import OutputExtension, ResultDirectoryExtension
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.experiments.output.sinks.sink_text import TextFileSink
from mlrl.testbed.extensions import Extension

from mlrl.util.cli import Argument, BoolArgument


class RuleModelAsTextExtension(Extension):
    """
    An extension that configures the functionality to write rule models to one or several sinks.
    """

    PRINT_RULES = BoolArgument(
        '--print-rules',
        description='Whether the induced rules should be printed on the console or not.',
        true_options={
            RuleModelAsText.OPTION_PRINT_FEATURE_NAMES, RuleModelAsText.OPTION_PRINT_OUTPUT_NAMES,
            RuleModelAsText.OPTION_PRINT_NOMINAL_VALUES, RuleModelAsText.OPTION_PRINT_BODIES,
            RuleModelAsText.OPTION_PRINT_HEADS, RuleModelAsText.OPTION_DECIMALS_BODY,
            RuleModelAsText.OPTION_DECIMALS_HEAD
        },
    )

    SAVE_RULES = BoolArgument(
        '--save-rules',
        description='Whether the induced rules should be written to a text file or not.',
        true_options={
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
    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.PRINT_RULES, self.SAVE_RULES}

    def __create_log_sinks(self, args: Namespace) -> List[Sink]:
        value, options = self.PRINT_RULES.get_value(args, default=OutputArguments.PRINT_ALL.get_value(args))

        if value:
            return [LogSink(options)]
        return []

    def __create_text_file_sinks(self, args: Namespace) -> List[Sink]:
        value, options = self.SAVE_RULES.get_value(args, default=OutputArguments.SAVE_ALL.get_value(args))
        result_directory = ResultDirectoryArguments.RESULT_DIR.get_value(args)

        if value and result_directory:
            return [
                TextFileSink(directory=Path(result_directory),
                             create_directory=OutputArguments.CREATE_DIRS.get_value(args),
                             options=options)
            ]
        return []

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_text_file_sinks(args)

        if sinks:
            writer = RuleModelAsTextWriter().add_sinks(*sinks)
            experiment_builder.add_post_training_output_writers(writer)
