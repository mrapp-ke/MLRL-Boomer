"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement a mode of operation for performing multiple experiments.
"""
from argparse import Namespace
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, List, override

import yaml

from mlrl.testbed.experiments import Experiment
from mlrl.testbed.modes.mode import Mode
from mlrl.testbed.util.io import open_readable_file

from mlrl.util.cli import CommandLineInterface, FlagArgument, StringArgument


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

    @override
    def configure_arguments(self, cli: CommandLineInterface):
        cli.add_arguments(self.CONFIG_FILE, self.LIST_COMMANDS)

    @override
    def run_experiment(self, _: Namespace, experiment_builder_factory: Experiment.Builder.Factory):
        pass
