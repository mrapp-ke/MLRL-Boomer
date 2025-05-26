"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write rule models to one or several sinks.
"""
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Set

from mlrl.testbed.experiments import Experiment
from mlrl.testbed.experiments.output.model_text import RuleModelAsText
from mlrl.testbed.experiments.output.model_text.extractor_rules import RuleModelAsTextExtractor
from mlrl.testbed.experiments.output.model_text.writer import ModelAsTextWriter
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.experiments.output.sinks.sink_text import TextFileSink
from mlrl.testbed.extensions import Extension

from mlrl.util.format import format_set
from mlrl.util.options import BooleanOption, parse_param_and_options


class RuleModelExtension(Extension):
    """
    An extension that configures the functionality to write rule models to one or several sinks.
    """

    PARAM_PRINT_RULES = '--print-rules'

    PRINT_RULES_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            RuleModelAsText.OPTION_PRINT_FEATURE_NAMES, RuleModelAsText.OPTION_PRINT_OUTPUT_NAMES,
            RuleModelAsText.OPTION_PRINT_NOMINAL_VALUES, RuleModelAsText.OPTION_PRINT_BODIES,
            RuleModelAsText.OPTION_PRINT_HEADS, RuleModelAsText.OPTION_DECIMALS_BODY,
            RuleModelAsText.OPTION_DECIMALS_HEAD
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_RULES = '--store-rules'

    STORE_RULES_VALUES = PRINT_RULES_VALUES

    def configure_arguments(self, argument_parser: ArgumentParser):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_arguments`
        """
        argument_parser.add_argument(
            self.PARAM_PRINT_RULES,
            type=str,
            default=BooleanOption.FALSE.value,
            help='Whether the induced rules should be printed on the console or not. Must be one of '
            + format_set(self.PRINT_RULES_VALUES.keys()) + '. For additional options refer to the documentation.')
        argument_parser.add_argument(
            self.PARAM_STORE_RULES,
            type=str,
            default=BooleanOption.FALSE.value,
            help='Whether the induced rules should be written into a text file or not. Must be one of '
            + format_set(self.STORE_RULES_VALUES.keys()) + '. For additional options refer to the documentation.')

    def __create_log_sinks(self, args: Namespace) -> List[Sink]:
        value, options = parse_param_and_options(self.PARAM_PRINT_RULES, args.print_rules, self.PRINT_RULES_VALUES)

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            return [LogSink(options)]
        return []

    def __create_text_file_sinks(self, args: Namespace) -> List[Sink]:
        value, options = parse_param_and_options(self.PARAM_STORE_RULES, args.store_rules, self.STORE_RULES_VALUES)

        if ((not value and args.store_all) or value == BooleanOption.TRUE.value) and args.output_dir:
            return [TextFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options)]
        return []

    def configure_experiment(self, args: Namespace, experiment: Experiment):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_text_file_sinks(args)

        if sinks:
            writer = ModelAsTextWriter(RuleModelAsTextExtractor(), exit_on_error=args.exit_on_error).add_sinks(*sinks)
            experiment.add_post_training_output_writers(writer)
