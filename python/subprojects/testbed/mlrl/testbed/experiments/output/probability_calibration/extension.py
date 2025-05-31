"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write calibration models to one or several sinks.
"""
from argparse import Namespace
from typing import Dict, List, Set

from mlrl.testbed.cli import Argument, BoolArgument
from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.probability_calibration.extractor_rules import \
    IsotonicJointProbabilityCalibrationModelExtractor, IsotonicMarginalProbabilityCalibrationModelExtractor
from mlrl.testbed.experiments.output.probability_calibration.writer import ProbabilityCalibrationModelWriter
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.util.format import OPTION_DECIMALS

from mlrl.util.options import BooleanOption, parse_param_and_options


class MarginalProbabilityCalibrationModelExtension(Extension):
    """
    An extension that configures the functionality to write models for the calibration of marginal probabilities to
    outputs.
    """

    PARAM_PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL = '--print-marginal-probability-calibration-model'

    PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_DECIMALS},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL = '--store-marginal-probability-calibration-model'

    STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES = PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES

    def get_arguments(self) -> List[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_arguments`
        """
        return [
            BoolArgument(
                self.PARAM_PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                default=False,
                help='Whether the model for the calibration of marginal probabilities should be printed on the console '
                + 'or not.',
                true_options={OPTION_DECIMALS},
            ),
            BoolArgument(
                self.PARAM_STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                default=False,
                help='Whether the model for the calibration of marginal probabilities should be written into an output '
                + 'file or not. Does only have an effect if the argument ' + OutputExtension.PARAM_OUTPUT_DIR + ' is '
                + 'specified.',
                true_options={OPTION_DECIMALS},
            ),
        ]

    def __create_log_sinks(self, args: Namespace) -> List[Sink]:
        value, options = parse_param_and_options(self.PARAM_PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                                                 args.print_marginal_probability_calibration_model,
                                                 self.PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES)
        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            return [LogSink(options)]
        return []

    def __create_csv_file_sinks(self, args: Namespace) -> List[Sink]:
        value, options = parse_param_and_options(self.PARAM_STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                                                 args.store_marginal_probability_calibration_model,
                                                 self.STORE_MARGINAL_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if ((not value and args.store_all) or value == BooleanOption.TRUE.value) and args.output_dir:
            return [CsvFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options)]
        return []

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_csv_file_sinks(args)

        if sinks:
            writer = ProbabilityCalibrationModelWriter(
                IsotonicMarginalProbabilityCalibrationModelExtractor()).add_sinks(*sinks)
            experiment_builder.add_post_training_output_writers(writer)


class JointProbabilityCalibrationModelExtension(Extension):
    """
    An extension that configures the functionality to write models for the calibration of joint probabilities to
    outputs.
    """

    PARAM_PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL = '--print-joint-probability-calibration-model'

    PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_DECIMALS},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_JOINT_PROBABILITY_CALIBRATION_MODEL = '--store-joint-probability-calibration-model'

    STORE_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES = PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES

    def get_arguments(self) -> List[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_arguments`
        """
        return [
            BoolArgument(
                self.PARAM_PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL,
                default=False,
                help='Whether the model for the calibration of joint probabilities should be printed on the console or '
                + 'not.',
                true_options={OPTION_DECIMALS},
            ),
            BoolArgument(
                self.PARAM_STORE_JOINT_PROBABILITY_CALIBRATION_MODEL,
                default=False,
                help='Whether the model for the calibration of joint probabilities should be written into an output '
                + 'file or not. Does only have an effect if the argument ' + OutputExtension.PARAM_OUTPUT_DIR + ' is '
                + 'specified.',
                true_options={OPTION_DECIMALS},
            ),
        ]

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = []
        value, options = parse_param_and_options(self.PARAM_PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL,
                                                 args.print_joint_probability_calibration_model,
                                                 self.PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            sinks.append(LogSink(options))

        value, options = parse_param_and_options(self.PARAM_STORE_JOINT_PROBABILITY_CALIBRATION_MODEL,
                                                 args.store_joint_probability_calibration_model,
                                                 self.STORE_JOINT_PROBABILITY_CALIBRATION_MODEL_VALUES)

        if ((not value and args.store_all) or value == BooleanOption.TRUE.value) and args.output_dir:
            sinks.append(
                CsvFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options))

        if sinks:
            writer = ProbabilityCalibrationModelWriter(
                IsotonicJointProbabilityCalibrationModelExtractor()).add_sinks(*sinks)
            experiment_builder.add_post_training_output_writers(writer)
