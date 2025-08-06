"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write calibration models to one or several sinks.
"""
from argparse import Namespace
from pathlib import Path
from typing import List, Set, override

from mlrl.boosting.testbed.experiments.output.probability_calibration.writer import \
    JointProbabilityCalibrationModelWriter, MarginalProbabilityCalibrationModelWriter

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.arguments import OutputArguments, ResultDirectoryArguments
from mlrl.testbed.experiments.output.extension import OutputExtension, ResultDirectoryExtension
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.util.format import OPTION_DECIMALS

from mlrl.util.cli import Argument, BoolArgument


class MarginalProbabilityCalibrationModelExtension(Extension):
    """
    An extension that configures the functionality to write models for the calibration of marginal probabilities to
    outputs.
    """

    PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL = BoolArgument(
        '--print-marginal-probability-calibration-model',
        description='Whether the model for the calibration of marginal probabilities should be printed on the console '
        + 'or not.',
        true_options={OPTION_DECIMALS},
    )

    SAVE_MARGINAL_PROBABILITY_CALIBRATION_MODEL = BoolArgument(
        '--save-marginal-probability-calibration-model',
        description='Whether the model for the calibration of marginal probabilities should be written to an output '
        + 'file or not.',
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
        return {self.PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL, self.SAVE_MARGINAL_PROBABILITY_CALIBRATION_MODEL}

    def __create_log_sinks(self, args: Namespace) -> List[Sink]:
        value, options = self.PRINT_MARGINAL_PROBABILITY_CALIBRATION_MODEL.get_value(
            args, default=OutputArguments.PRINT_ALL.get_value(args))

        if value:
            return [LogSink(options)]
        return []

    def __create_csv_file_sinks(self, args: Namespace) -> List[Sink]:
        value, options = self.SAVE_MARGINAL_PROBABILITY_CALIBRATION_MODEL.get_value(
            args, default=OutputArguments.SAVE_ALL.get_value(args))
        result_directory = ResultDirectoryArguments.RESULT_DIR.get_value(args)

        if value and result_directory:
            return [
                CsvFileSink(directory=Path(result_directory),
                            create_directory=OutputArguments.CREATE_DIRS.get_value(args),
                            options=options)
            ]
        return []

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_csv_file_sinks(args)

        if sinks:
            writer = MarginalProbabilityCalibrationModelWriter().add_sinks(*sinks)
            experiment_builder.add_post_training_output_writers(writer)


class JointProbabilityCalibrationModelExtension(Extension):
    """
    An extension that configures the functionality to write models for the calibration of joint probabilities to
    outputs.
    """

    PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL = BoolArgument(
        '--print-joint-probability-calibration-model',
        description='Whether the model for the calibration of joint probabilities should be printed on the console or '
        + 'not.',
        true_options={OPTION_DECIMALS},
    )

    SAVE_JOINT_PROBABILITY_CALIBRATION_MODEL = BoolArgument(
        '--save-joint-probability-calibration-model',
        description='Whether the model for the calibration of joint probabilities should be written to an output file '
        + 'or not.',
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
        return {self.PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL, self.SAVE_JOINT_PROBABILITY_CALIBRATION_MODEL}

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = []
        value, options = self.PRINT_JOINT_PROBABILITY_CALIBRATION_MODEL.get_value(
            args, default=OutputArguments.PRINT_ALL.get_value(args))

        if value:
            sinks.append(LogSink(options))

        value, options = self.SAVE_JOINT_PROBABILITY_CALIBRATION_MODEL.get_value(
            args, default=OutputArguments.SAVE_ALL.get_value(args))
        result_directory = ResultDirectoryArguments.RESULT_DIR.get_value(args)

        if value and result_directory:
            sinks.append(
                CsvFileSink(directory=Path(result_directory),
                            create_directory=OutputArguments.CREATE_DIRS.get_value(args),
                            options=options))

        if sinks:
            writer = JointProbabilityCalibrationModelWriter().add_sinks(*sinks)
            experiment_builder.add_post_training_output_writers(writer)
