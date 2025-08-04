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
from typing import Any, Callable, Generator, Iterable, List, Optional, override

import yamale

from mlrl.testbed.command import ArgumentDict, ArgumentList, Command
from mlrl.testbed.experiments.fold import FoldingStrategy
from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.experiments.timer import Timer
from mlrl.testbed.modes.mode import Mode

from mlrl.util.cli import BoolArgument, CommandLineInterface, FlagArgument, SetArgument, StringArgument
from mlrl.util.format import format_iterable
from mlrl.util.options import Options

Batch = List[Command]


class BatchMode(Mode):
    """
    An abstract base class for all modes of operation that perform multiple experiments.
    """

    class Runner(ABC):
        """"
        An abstract base class for all classes that allow to run experiments.
        """

        def __init__(self, name: str):
            """
            :param name: The name of the runner
            """
            self.name = name

        @abstractmethod
        def run_batch(self, args: Namespace, batch: Batch, recipe: Recipe):
            """
            Must be implemented by subclasses in order to run all commands in a given batch.

            :param args:    The command line arguments specified by the user
            :param batch:   The batch of experiments to be run
            :param recipe:  A `Recipe` that provides access to the ingredients that are needed for setting up
                            experiments
            """

    class LogRunner(Runner):
        """
        Allows to log the commands of experiments, instead of executing them.
        """

        @staticmethod
        def __format_command(command: Iterable[str]) -> str:
            formatted_command = ''

            for i, argument in enumerate(command):
                if i > 0:
                    formatted_command += ' '

                    if argument.startswith('-'):
                        formatted_command += '\\\n    '

                formatted_command += regex.sub(r'({.*})', '\'\\1\'', argument)  # Escape curly braces with single quotes

            return formatted_command

        def __init__(self):
            super().__init__(name='log')

        @override
        def run_batch(self, args: Namespace, batch: Batch, recipe: Recipe):
            for i, command in enumerate(batch):
                recipe.create_experiment_builder(command.apply_to_namespace(args))

                if i > 0:
                    log.info('')

                log.info('%s', self.__format_command(command))

    class SequentialRunner(Runner):
        """
        Allows to run experiments sequentially.
        """

        def __init__(self):
            super().__init__(name='sequential')

        @override
        def run_batch(self, args: Namespace, batch: Batch, recipe: Recipe):
            start_time = Timer.start()

            namespaces = [command.apply_to_namespace(args) for command in batch]
            num_experiments = len(batch)

            for experiment_namespace in namespaces:
                recipe.create_experiment_builder(experiment_namespace)  # For validation

            for i, (experiment_namespace, command) in enumerate(zip(namespaces, batch)):
                if i == 0:
                    log.info('Running %s %s...', num_experiments,
                             'experiments' if num_experiments > 1 else 'experiment')

                log.info('\nRunning experiment (%s / %s): "%s"', i + 1, num_experiments,
                         format_iterable(command, separator=' '))
                recipe.create_experiment_builder(experiment_namespace).run()

            run_time = Timer.stop(start_time)
            log.info('Successfully finished %s %s after %s', num_experiments,
                     'experiments' if num_experiments > 1 else 'experiment', run_time)

    class ConfigFile(ABC):
        """
        A YAML configuration file that configures the batch of experiments to be run.
        """

        Factory = Callable[[str], 'BatchMode.ConfigFile']

        @dataclass
        class Parameter:
            """
            A parameter defined in the configuration file.

            Attributes:
                name:   The name of the parameter
                values: One or several values that should be set for the parameter
            """
            name: str
            values: List['BatchMode.ConfigFile.ParameterValue']

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
                return [BatchMode.ConfigFile.ParameterValue(value=values)]

            parameter_values = []

            for value in values:
                if isinstance(value, str):
                    parameter_values.append(BatchMode.ConfigFile.ParameterValue(value=value))
                else:
                    parameter_values.append(
                        BatchMode.ConfigFile.ParameterValue(value=value['value'],
                                                            additional_arguments=value.get('additional_arguments', [])))

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
                parameters.append(BatchMode.ConfigFile.Parameter(name=parameter_dict['name'], values=parameter_values))

            return parameters

        @cached_property
        def parameter_args(self) -> List[ArgumentList]:
            """
            A list that contains the command line arguments corresponding to the algorithmic parameters to be used in
            the different experiments defined in the configuration file.
            """
            parameter_args: List[ArgumentList] = [ArgumentList()]

            for parameter in self.parameters:
                updated_parameter_args = []

                for argument_list in parameter_args:
                    for parameter_value in parameter.values:
                        additional_arguments: List[str] = []
                        additional_arguments = reduce(lambda aggr, arguments: aggr + arguments.split(),
                                                      parameter_value.additional_arguments, additional_arguments)
                        arguments = argument_list + [parameter.name, parameter_value.value] + additional_arguments
                        updated_parameter_args.append(ArgumentList(arguments))

                parameter_args = updated_parameter_args

            return parameter_args

        @property
        @abstractmethod
        def dataset_args(self) -> List[ArgumentList]:
            """
            A list that contains the command line arguments corresponding to the datasets to be used in the different
            experiments defined in the configuration file.
            """

        @override
        def __str__(self) -> str:
            return self.file_path

    CONFIG_FILE = StringArgument(
        '--config',
        required=True,
        description='An absolute or relative path to a YAML file that configures the batch of experiments to be run.',
    )

    SEPARATE_FOLDS = BoolArgument(
        '--separate-folds',
        default=True,
        description='Whether separate experiments should be run for the individual folds of a cross validation or not.',
    )

    LIST_COMMANDS = FlagArgument(
        '--list',
        description='Lists the commands for running individual experiments instead of executing them.',
    )

    RUNNER = SetArgument(
        '--runner',
        values={},
        description='The runner to be used for running individual experiments in a batch.',
    )

    @property
    def runners(self) -> List[Runner]:
        """
        The runners that may be used for running individual experiments in a batch.
        """
        return [BatchMode.SequentialRunner()] + self.runners_

    def __create_runner_argument(self) -> SetArgument:
        runners = self.runners
        return SetArgument(
            *self.RUNNER.names,
            values={runner.name
                    for runner in runners},
            default=runners[0].name,
            description=self.RUNNER.description,
        )

    def __process_commands(self, namespace: Namespace, config_file: ConfigFile, recipe: Recipe):
        separate_folds = self.SEPARATE_FOLDS.get_value(namespace)
        batch = list(self.__get_commands(namespace, config_file, recipe, separate_folds=separate_folds))
        runner = self.__get_runner(namespace)
        runner.run_batch(namespace, batch, recipe)

    def __get_runner(self, namespace: Namespace) -> 'BatchMode.Runner':
        if self.LIST_COMMANDS.get_value(namespace):
            return BatchMode.LogRunner()

        runner_argument = self.__create_runner_argument()
        runner_name = runner_argument.get_value(namespace)
        runners_by_name = {runner.name: runner for runner in self.runners}
        return runners_by_name[runner_name]

    @staticmethod
    def __get_commands(namespace: Namespace,
                       config_file: ConfigFile,
                       recipe: Recipe,
                       separate_folds: bool = False) -> Generator[Command, None, None]:
        module_name = sys.argv[1]
        default_args = BatchMode.__filter_arguments(ArgumentList(sys.argv[2:]))

        for dataset_args in map(BatchMode.__filter_arguments, config_file.dataset_args):
            dataset_name = dataset_args['--dataset']

            if not dataset_name:
                raise RuntimeError('Unable to determine dataset name based on the arguments ' + str(dataset_args))

            for parameter_args in map(BatchMode.__filter_arguments, config_file.parameter_args):
                output_dir = BatchMode.__get_output_dir(parameter_args, dataset_name)
                argument_dict = ArgumentDict(
                    default_args | dataset_args | parameter_args | {
                        '--result-dir': str(output_dir / 'results'),
                        '--model-save-dir': str(output_dir / 'models'),
                        '--parameter-save-dir': str(output_dir / 'parameters'),
                    })
                command = Command(module_name=module_name, argument_list=argument_dict.to_list())

                if separate_folds:
                    dataset_splitter = recipe.create_dataset_splitter(command.apply_to_namespace(namespace))
                    folding_strategy = dataset_splitter.folding_strategy

                    if folding_strategy.is_cross_validation_used:
                        yield from BatchMode.__separate_folds(folding_strategy, module_name, argument_dict)
                    else:
                        yield command
                else:
                    yield command

    @staticmethod
    def __separate_folds(folding_strategy: FoldingStrategy, module_name: str,
                         argument_dict: ArgumentDict) -> Generator[Command, None, None]:
        options = Options({'num_folds': folding_strategy.num_folds})

        for fold in folding_strategy.folds:
            options.dictionary['first_fold'] = fold.index + 1
            options.dictionary['last_fold'] = fold.index + 1
            argument_dict_per_fold = ArgumentDict(argument_dict | {'--data-split': 'cross-validation' + str(options)})
            yield Command(module_name=module_name, argument_list=argument_dict_per_fold.to_list())

    @staticmethod
    def __get_output_dir(argument_dict: ArgumentDict, dataset_name: str) -> Path:
        return Path(
            *map(lambda argument: argument[0].lstrip('-') + ('_' + argument[1]) if argument[1] else '',
                 argument_dict.items()), 'dataset_' + dataset_name)

    @staticmethod
    def __filter_arguments(argument_list: ArgumentList) -> ArgumentDict:
        return argument_list.filter(*BatchMode.CONFIG_FILE.names, *Mode.MODE.names,
                                    BatchMode.LIST_COMMANDS.name).to_dict()

    def __init__(self, config_file_factory: Optional[ConfigFile.Factory] = None):
        """
        :param config_file_factory: A factory that allows to create the configuration file that configures the batch of
                                    experiments to be run or None, if no such factory is available
        """
        self.config_file_factory_ = config_file_factory
        self.runners_: List[BatchMode.Runner] = []

    def add_runner(self, runner: 'BatchMode.Runner') -> 'BatchMode':
        """
        Adds a runner that may be used in batch mode for running individual experiments in a batch.

        :param runner:  The runner to be added
        :return:        The batch mode itself
        """
        self.runners_.append(runner)
        return self

    @override
    def configure_arguments(self, cli: CommandLineInterface):
        cli.add_arguments(self.CONFIG_FILE, self.SEPARATE_FOLDS, self.LIST_COMMANDS, self.__create_runner_argument())

    @override
    def run_experiment(self, args: Namespace, recipe: Recipe):
        config_file_path = self.CONFIG_FILE.get_value(args)

        if config_file_path:
            config_file_factory = self.config_file_factory_
            config_file = config_file_factory(config_file_path) if config_file_factory else None

            if config_file:
                self.__process_commands(args, config_file, recipe)
