"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement a mode of operation for performing multiple experiments.
"""
import logging as log
import re as regex
import sys

from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass, field
from functools import cached_property, reduce
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, override

import yamale

from mlrl.testbed.experiments import Experiment
from mlrl.testbed.experiments.timer import Timer
from mlrl.testbed.modes.mode import Mode

from mlrl.util.cli import CommandLineInterface, FlagArgument, StringArgument
from mlrl.util.format import format_iterable


class BatchExperimentMode(Mode):
    """
    An abstract base class for all modes of operation that perform multiple experiments.
    """

    class ConfigFile(ABC):
        """
        A YAML configuration file that configures the batch of experiments to be run.
        """

        Factory = Callable[[str], 'BatchExperimentMode.ConfigFile']

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

        def __init__(self, file_path: str, schema_file_path: str):
            """
            :param file_path: The path to the configuration file
            """
            schema = yamale.make_schema(schema_file_path)
            data = yamale.make_data(file_path)
            yamale.validate(schema, data)
            self.file_path = file_path
            self.yaml_dict = data[0][0]

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

        @property
        def parameter_args(self) -> List[List[str]]:
            """
            A list that contains the command line arguments corresponding to the algorithmic parameters to be used in
            the different experiments defined in the configuration file.
            """
            parameter_args: List[List[str]] = [[]]

            for parameter in self.parameters:
                updated_parameter_args = []

                for args in parameter_args:
                    for parameter_value in parameter.values:
                        additional_arguments: List[str] = []
                        additional_arguments = reduce(lambda aggr, arguments: aggr + arguments.split(),
                                                      parameter_value.additional_arguments, additional_arguments)
                        updated_parameter_args.append(args + [parameter.name, parameter_value.value]
                                                      + additional_arguments)

                parameter_args = updated_parameter_args

            return parameter_args

        @property
        @abstractmethod
        def dataset_args(self) -> List[List[str]]:
            """
            A list that contains the command line arguments corresponding to the datasets to be used in the different
            experiments defined in the configuration file.
            """

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
            formatted_command = ''

            for i, argument in enumerate(command):
                if i > 0:
                    formatted_command += ' '

                    if argument.startswith('-'):
                        formatted_command += '\\\n    '

                formatted_command += regex.sub(r'({.*})', '\'\\1\'', argument)  # Escape curly braces with single quotes

            log.info('%s\n', formatted_command)

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
        default_args = BatchExperimentMode.__filter_and_parse_args(sys.argv[2:])
        dataset_batch_args = (BatchExperimentMode.__filter_and_parse_args(dataset_args)
                              for dataset_args in config_file.dataset_args)
        parameter_batch_args = (BatchExperimentMode.__filter_and_parse_args(parameter_args)
                                for parameter_args in config_file.parameter_args)
        args = []

        for dataset_args in dataset_batch_args:
            dataset_name = dataset_args['--dataset']

            if not dataset_name:
                raise RuntimeError('Unable to determine dataset name based on the arguments ' + str(dataset_args))

            for parameter_args in parameter_batch_args:
                module_name = sys.argv[1]
                output_dir = BatchExperimentMode.__get_output_dir(parameter_args, dataset_name)
                arg_list = [
                    module_name,
                    '--result-dir',
                    str(output_dir / 'results'),
                    '--model-save-dir',
                    str(output_dir / 'models'),
                    '--parameter-save-dir',
                    str(output_dir / 'parameters'),
                ]

                args_dict = default_args | dataset_args | parameter_args

                for key in sorted(args_dict.keys()):
                    arg_list.append(key)
                    value = args_dict.get(key)

                    if value:
                        arg_list.append(value)

                args.append(arg_list)

        return args

    @staticmethod
    def __get_output_dir(parameters: Dict[str, Optional[str]], dataset_name: str) -> Path:
        return Path(
            *map(lambda parameter: parameter[0].lstrip('-') + ('_' + parameter[1]) if parameter[1] else '',
                 parameters.items()), 'dataset_' + dataset_name)

    @staticmethod
    def __filter_and_parse_args(args: List[str]) -> Dict[str, Optional[str]]:
        return BatchExperimentMode.__parse_args(
            BatchExperimentMode.__filter_args(args, *BatchExperimentMode.CONFIG_FILE.names, *Mode.MODE.names,
                                              BatchExperimentMode.LIST_COMMANDS.name), )

    @staticmethod
    def __filter_args(args: List[str], *ignored_args: str) -> List[str]:
        ignored_args_set = set(ignored_args)
        filtered_args = []
        skip = False

        for arg in args:
            skip = (skip and not arg.startswith('-')) or arg in ignored_args_set

            if not skip:
                filtered_args.append(arg)

        return filtered_args

    @staticmethod
    def __parse_args(args: List[str]) -> Dict[str, Optional[str]]:
        args_dict: Dict[str, Optional[str]] = {}
        previous_arg = None

        for arg in args:
            if not previous_arg or arg.startswith('-'):
                args_dict.setdefault(arg, None)
                previous_arg = arg
            else:
                args_dict[previous_arg] = arg
                previous_arg = None

        return args_dict

    @staticmethod
    def __add_args_to_namespace(namespace: Namespace, args: List[str]) -> Namespace:
        for i in range(1, len(args)):
            arg = args[i]

            if arg.startswith('-'):
                argument_name = arg.lstrip('-').replace('-', '_')
                argument_value = None

                if i + 1 < len(args):
                    next_arg = args[i + 1]

                    if not next_arg.startswith('-'):
                        argument_value = next_arg

                setattr(namespace, argument_name, argument_value if argument_value else True)

        return namespace

    def __init__(self, config_file_factory: Optional[ConfigFile.Factory]):
        """
        :param config_file_factory: A factory that allows to create the configuration file that configures the batch of
                                    experiments to be run or None, if no such factory is available
        """
        self.config_file_factory_ = config_file_factory

    @override
    def configure_arguments(self, cli: CommandLineInterface):
        cli.add_arguments(self.CONFIG_FILE, self.LIST_COMMANDS)

    @override
    def run_experiment(self, args: Namespace, experiment_builder_factory: Experiment.Builder.Factory):
        config_file_path = self.CONFIG_FILE.get_value(args)

        if config_file_path:
            config_file_factory = self.config_file_factory_
            config_file = config_file_factory(config_file_path) if config_file_factory else None

            if config_file:
                self.__process_commands(args, config_file, experiment_builder_factory)
