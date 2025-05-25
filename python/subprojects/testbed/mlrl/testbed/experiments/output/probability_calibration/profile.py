"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write calibration models to outputs.
"""
from argparse import ArgumentParser, Namespace
from typing import Dict, Set

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.probability_calibration.extractor_rules import \
    IsotonicMarginalProbabilityCalibrationModelExtractor
from mlrl.testbed.experiments.output.probability_calibration.writer import ProbabilityCalibrationModelWriter
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.profiles.profile import Profile
from mlrl.testbed.util.format import OPTION_DECIMALS

from mlrl.util.format import format_enum_values
from mlrl.util.options import BooleanOption, parse_param_and_options


class MarginalProbabilityCalibrationModelProfile(Profile):
    """
    A profile that configures the functionality to write models for the calibration of marginal probabilities to
    outputs.
    """

    PARAM_PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL = '--print-marginal-probability-calibration-model'

    PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_DECIMALS},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL = '--store-marginal-probability-calibration-model'

    STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES = PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES

    def configure_arguments(self, argument_parser: ArgumentParser):
        """
        See :func:`mlrl.testbed.profiles.profile.Profile.configure_arguments`
        """
        argument_parser.add_argument(
            self.PARAM_PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
            type=str,
            default=BooleanOption.FALSE.value,
            help='Whether the model for the calibration of marginal probabilities should be printed on the console or '
            + 'not. Must be one of ' + format_enum_values(BooleanOption) + '. For additional options refer to the '
            + 'documentation.')
        argument_parser.add_argument(
            self.PARAM_STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
            type=str,
            default=BooleanOption.FALSE.value,
            help='Whether the model for the calibration of marginal probabilities should be written into an output '
            + 'file or not. Must be one of ' + format_enum_values(BooleanOption) + '. For additional options '
            + 'refer to the documentation.')

    def configure_experiment(self, args: Namespace, experiment: Experiment):
        """
        See :func:`mlrl.testbed.profiles.profile.Profile.configure_experiment`
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                                                 args.print_marginal_probability_calibration_model,
                                                 self.PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            sinks.append(LogSink(options))

        value, options = parse_param_and_options(self.PARAM_STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                                                 args.store_marginal_probability_calibration_model,
                                                 self.STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if ((not value and args.store_all) or value == BooleanOption.TRUE.value) and args.output_dir:
            sinks.append(
                CsvFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options))

        if sinks:
            writer = ProbabilityCalibrationModelWriter(IsotonicMarginalProbabilityCalibrationModelExtractor(),
                                                       exit_on_error=args.exit_on_error).add_sinks(*sinks)
            experiment.add_post_training_output_writers(writer)
