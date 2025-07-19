"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement a mode of operation for performing multiple experiments.
"""
import logging as log
import sys

from argparse import Namespace
from dataclasses import dataclass, field
from functools import cached_property, reduce
from typing import Any, Dict, List, override

import yaml

from mlrl.testbed.experiments import Experiment
from mlrl.testbed.experiments.timer import Timer
from mlrl.testbed.modes.mode import Mode
from mlrl.testbed.util.io import open_readable_file

from mlrl.util.cli import CommandLineInterface, FlagArgument, StringArgument
from mlrl.util.format import format_iterable


class BatchExperimentMode(Mode):
    """
    A mode of operation that performs multiple experiments.
    """

    class ConfigFile:
        """
        A YAML configuration file that configures the batch of experiments to be run.
        """

        @dataclass
        class Parameter:
            """
            A parameter defined in the configuration file.

            Attributes:
                name:   The name of the parameter
                values: One or several values that should be set for the parameter
            """
            name: str
            values: List['BatchExperimentMode.ConfigFile.ParameterValue']

        @dataclass
        class ParameterValue:
            """
            A value that should be set for a parameter.

            Attributes:
                value:                  The value to be set
                additional_arguments:   Additional arguments associated with the parameter
            """
            value: str
            additional_arguments: List[str] = field(default_factory=list)

        @staticmethod
        def __parse_parameter_values(values: List[Any]) -> List[ParameterValue]:
            if isinstance(values, str):
                return [BatchExperimentMode.ConfigFile.ParameterValue(value=values)]

            parameter_values = []

            for value in values:
                if isinstance(value, str):
                    parameter_values.append(BatchExperimentMode.ConfigFile.ParameterValue(value=value))
                else:
                    parameter_values.append(
                        BatchExperimentMode.ConfigFile.ParameterValue(
                            value=value['value'], additional_arguments=value['additional_arguments']))

            return parameter_values

        def __init__(self, file_path: str):
            """
            :param file_path: The path to the configuration file
            """
            self.file_path = file_path

        @cached_property
        def yaml_dict(self) -> Dict[Any, Any]:
            """
            A dictionary that contains the content of the configuration file.
            """
            with open_readable_file(self.file_path) as file:
                yaml_dict = yaml.safe_load(file)
                return yaml_dict if yaml_dict else {}

        @cached_property
        def parameters(self) -> List[Parameter]:
            """
            A list that contains all parameters defined in the configuration file.
            """
            parameters = []

            for parameter_dict in self.yaml_dict.get('parameters', []):
                parameter_values = self.__parse_parameter_values(parameter_dict['values'])
                parameters.append(
                    BatchExperimentMode.ConfigFile.Parameter(name=parameter_dict['name'], values=parameter_values))

            return parameters

        @override
        def __str__(self) -> str:
            return self.file_path

    CONFIG_FILE = StringArgument(
        '-c',
        '--config',
        required=True,
        description='An absolute or relative path to a YAML file that configures the batch of experiments to be run.',
    )

    LIST_COMMANDS = FlagArgument(
        '--list',
        description='Lists the commands for running individual experiments instead of executing them.',
    )

    def __process_commands(self, args: Namespace, config_file: ConfigFile,
                           experiment_builder_factory: Experiment.Builder.Factory):
        if self.LIST_COMMANDS.get_value(args):
            self.__list_commands(args, config_file, experiment_builder_factory)
        else:
            self.__run_commands(args, config_file, experiment_builder_factory)

    @staticmethod
    def __list_commands(namespace: Namespace, config_file: ConfigFile,
                        experiment_builder_factory: Experiment.Builder.Factory):
        for arguments in BatchExperimentMode.__get_args(config_file):
            experiment_builder_factory(BatchExperimentMode.__add_args_to_namespace(namespace, arguments))
            command = ['mlrl-testbed'] + arguments
            log.info(format_iterable(command, separator=' '))

    @staticmethod
    def __run_commands(namespace: Namespace, config_file: ConfigFile,
                       experiment_builder_factory: Experiment.Builder.Factory):
        args = BatchExperimentMode.__get_args(config_file)
        start_time = Timer.start()

        for i, arguments in enumerate(args):
            command = ['mlrl-testbed'] + arguments
            experiment_builder = experiment_builder_factory(
                BatchExperimentMode.__add_args_to_namespace(namespace, arguments))

            if i == 0:
                log.info('Running %s %s...', len(args), 'experiments' if len(args) > 1 else 'experiment')

            log.info('Running experiment (%s/%s): "%s"', i + 1, len(args), format_iterable(command, separator=' '))
            experiment_builder.run()

        run_time = Timer.stop(start_time)
        log.info('Successfully finished %s %s after %s', len(args), 'experiments' if len(args) > 1 else 'experiment',
                 run_time)

    @staticmethod
    def __get_args(config_file: ConfigFile) -> List[List[str]]:
        args = []
        default_args = BatchExperimentMode.__get_default_args()

        for parameter_args in BatchExperimentMode.__get_parameter_args(config_file):
            args.append(default_args + parameter_args)

        return args

    @staticmethod
    def __get_default_args() -> List[str]:
        default_args = []
        skip_next = False

        for arg in sys.argv[1:]:
            if skip_next:
                skip_next = False
            elif arg in BatchExperimentMode.CONFIG_FILE.names:
                skip_next = True
            elif arg != BatchExperimentMode.LIST_COMMANDS.name:
                default_args.append(arg)

        return default_args

    @staticmethod
    def __get_parameter_args(config_file: ConfigFile) -> List[List[str]]:
        parameter_args: List[List[str]] = [[]]

        for parameter in config_file.parameters:
            updated_parameter_args = []

            for args in parameter_args:
                for parameter_value in parameter.values:
                    additional_arguments: List[str] = []
                    additional_arguments = reduce(lambda aggr, arguments: aggr + arguments.split(),
                                                  parameter_value.additional_arguments, additional_arguments)
                    updated_parameter_args.append(args + [parameter.name, parameter_value.value] + additional_arguments)

            parameter_args = updated_parameter_args

        return parameter_args

    @staticmethod
    def __add_args_to_namespace(namespace: Namespace, args: List[str]) -> Namespace:
        for i in range(1, len(args)):
            arg = args[i]

            if arg.startswith('-'):
                argument_name = arg.lstrip('--').replace('-', '_')
                argument_value = None

                if i + 1 < len(args):
                    next_arg = args[i + 1]

                    if not next_arg.startswith('-'):
                        argument_value = next_arg

                setattr(namespace, argument_name, argument_value if argument_value else True)

        return namespace

    @override
    def configure_arguments(self, cli: CommandLineInterface):
        cli.add_arguments(self.CONFIG_FILE, self.LIST_COMMANDS)

    @override
    def run_experiment(self, args: Namespace, experiment_builder_factory: Experiment.Builder.Factory):
        config_file_path = self.CONFIG_FILE.get_value(args)

        if config_file_path:
            config_file = BatchExperimentMode.ConfigFile(config_file_path)
            self.__process_commands(args, config_file, experiment_builder_factory)
