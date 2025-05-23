"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for programs that can be configured via command line arguments.
"""

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import List, Optional

from mlrl.testbed.experiments import Experiment
from mlrl.testbed.profiles import Profile
from mlrl.testbed.profiles.profile_log import LogProfile
from mlrl.testbed.program_info import ProgramInfo

from mlrl.util.format import format_enum_values
from mlrl.util.options import BooleanOption


class Runnable(ABC):
    """
    An abstract base class for all programs that can be configured via the command line API. The programs functionality
    is implemented by individual profiles that are applied to the runnable.
    """

    class BaseProfile(Profile):
        """
        A basic profile that is applied to all runnables.
        """

        def configure_arguments(self, argument_parser: ArgumentParser):
            """
            See :func:`mlrl.testbed.profiles.profile.Profile.configure_arguments`
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

        def configure_experiment(self, args: Namespace, _: Experiment):
            """
            See :func:`mlrl.testbed.profiles.profile.Profile.configure_experiment`
            """

    def run(self, args: Namespace):
        """
        Executes the runnable.

        :param args: The command line arguments specified by the user
        """
        experiment = self.create_experiment(args)

        for profile in self.get_profiles():
            profile.configure_experiment(args, experiment)

        should_predict = bool(experiment.prediction_output_writers)
        experiment.run(predict_for_training_dataset=should_predict and args.predict_for_training_data,
                       predict_for_test_dataset=should_predict and args.predict_for_test_data)

    def get_profiles(self) -> List[Profile]:
        """
        May be overridden by subclasses in order to return the profiles that should be applied to the runnable.

        :return: A list that contains the profiles to be applied to the runnable
        """
        return [Runnable.BaseProfile(), LogProfile()]

    def get_program_info(self) -> Optional[ProgramInfo]:
        """
        May be overridden by subclasses in order to provide information about the program to be printed via the command
        line argument '-v' or '--version'. 

        :return: A `ProgramInfo` or None, if no information is provided
        """
        return None

    # pylint: disable=unused-argument
    def configure_arguments(self, argument_parser: ArgumentParser, show_help: bool):
        """
        Configures a given argument parser according to the profiles applied to the runnable.

        :param argument_parser: The argument parser to be configured
        :param show_help:       True, if the help text of the program should be shown, False otherwise
        """
        # pylint: disable=assignment-from-none
        program_info = self.get_program_info()

        if program_info:
            argument_parser.add_argument('-v',
                                         '--version',
                                         action='version',
                                         version=str(program_info),
                                         help='Display information about the program\'s version.')

        for profile in self.get_profiles():
            profile.configure_arguments(argument_parser)

    @abstractmethod
    def create_experiment(self, args: Namespace) -> Experiment:
        """
        Must be implemented by subclasses in order to create the experiment to be run by the program.

        :param args:    The command line arguments specified by the user
        :return:        The experiment that has been created
        """
