"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for programs that can be configured via command line arguments.
"""

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import List, Optional

from mlrl.testbed.experiments import Experiment
from mlrl.testbed.extensions import Extension
from mlrl.testbed.extensions.extension_log import LogExtension
from mlrl.testbed.program_info import ProgramInfo

from mlrl.util.format import format_enum_values
from mlrl.util.options import BooleanOption


class Runnable(ABC):
    """
    An abstract base class for all programs that can be configured via the command line API. The programs functionality
    is implemented by individual extensions that are applied to the runnable.
    """

    class PredictionDatasetExtension(Extension):
        """
        An extension that configures the functionality to predict for different datasets.
        """

        def configure_arguments(self, argument_parser: ArgumentParser):
            """
            See :func:`mlrl.testbed.extensions.extension.Extension.configure_arguments`
            """
            argument_parser.add_argument(
                '--predict-for-training-data',
                type=BooleanOption.parse,
                default=False,
                help='Whether predictions should be obtained for the training data or not. Must be one of '
                + format_enum_values(BooleanOption) + '.')
            argument_parser.add_argument(
                '--predict-for-test-data',
                type=BooleanOption.parse,
                default=True,
                help='Whether predictions should be obtained for the test data or not. Must be one of '
                + format_enum_values(BooleanOption) + '.')

        def configure_experiment(self, args: Namespace, _: Experiment.Builder):
            """
            See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
            """

    class VersionExtension(Extension):
        """
        An extension that configures the functionality to show information about a program when the "--version" flag is
        passed to the command line API.
        """

        def __init__(self, program_info: Optional[ProgramInfo]):
            """
            :param program_info:    Information about a program to be shown when the "--version" flag is passed to the
                                    command line API
            """
            self.program_info = program_info

        def configure_arguments(self, argument_parser: ArgumentParser):
            """
            See :func:`mlrl.testbed.extensions.extension.Extension.configure_arguments`
            """
            program_info = self.program_info

            if program_info:
                argument_parser.add_argument('-v',
                                             '--version',
                                             action='version',
                                             version=str(program_info),
                                             help='Display information about the program\'s version.')

        def configure_experiment(self, args: Namespace, _: Experiment.Builder):
            """
            See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
            """

    def get_extensions(self) -> List[Extension]:
        """
        May be overridden by subclasses in order to return the extensions that should be applied to the runnable.

        :return: A list that contains the extensions to be applied to the runnable
        """
        return [
            Runnable.PredictionDatasetExtension(),
            Runnable.VersionExtension(self.get_program_info()),
            LogExtension()
        ]

    def get_program_info(self) -> Optional[ProgramInfo]:
        """
        May be overridden by subclasses in order to provide information about the program to be printed via the command
        line argument '-v' or '--version'. 

        :return: A `ProgramInfo` or None, if no information is provided
        """
        return None

    def run(self, args: Namespace):
        """
        Executes the runnable.

        :param args: The command line arguments specified by the user
        """
        experiment_builder = self.create_experiment_builder(args)

        for extension in self.get_extensions():
            extension.configure_experiment(args, experiment_builder)

        should_predict = bool(experiment_builder.prediction_output_writers)
        experiment = experiment_builder.build()
        experiment.run(predict_for_training_dataset=should_predict and args.predict_for_training_data,
                       predict_for_test_dataset=should_predict and args.predict_for_test_data)

    # pylint: disable=unused-argument
    def configure_arguments(self, argument_parser: ArgumentParser, show_help: bool):
        """
        Configures a given argument parser according to the extensions applied to the runnable.

        :param argument_parser: The argument parser to be configured
        :param show_help:       True, if the help text of the program should be shown, False otherwise
        """
        for extension in self.get_extensions():
            extension.configure_arguments(argument_parser)

    @abstractmethod
    def create_experiment_builder(self, args: Namespace) -> Experiment.Builder:
        """
        Must be implemented by subclasses in order to create the builder that allows to configure the experiment to be
        run by the program.

        :param args:    The command line arguments specified by the user
        :return:        The builder that has been created
        """
