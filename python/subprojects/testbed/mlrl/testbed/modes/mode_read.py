"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement a mode of operation for reading experimental results.
"""
import logging as log

from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, override

from mlrl.testbed_sklearn.experiments.output.dataset.arguments_ground_truth import GroundTruthArguments
from mlrl.testbed_sklearn.experiments.output.evaluation.evaluation_result import EvaluationResult

from mlrl.testbed.command import Command
from mlrl.testbed.experiments.dataset_type import DatasetType
from mlrl.testbed.experiments.experiment import Experiment, ExperimentalProcedure
from mlrl.testbed.experiments.input.data import InputData, TabularInputData
from mlrl.testbed.experiments.input.dataset.arguments import DatasetArguments
from mlrl.testbed.experiments.input.dataset.splitters.arguments import DatasetSplitterArguments
from mlrl.testbed.experiments.meta_data import MetaData
from mlrl.testbed.experiments.output.evaluation.evaluation_result import AggregatedEvaluationResult
from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.experiments.state import ExperimentMode, ExperimentState
from mlrl.testbed.experiments.table import Cell, RowWiseTable, Table
from mlrl.testbed.modes.mode import InputMode
from mlrl.testbed.modes.mode_batch import BatchMode

from mlrl.util.cli import Argument
from mlrl.util.options import Options


class ReadMode(InputMode):
    """
    An abstract base class for all modes of operation that read experimental results.
    """

    class SingleExperimentProcedure(ExperimentalProcedure):
        """
        The procedure that is used to conduct a single experiment in read mode.
        """

        @override
        def _before_experiment(self, experiment: Experiment, state: ExperimentState) -> ExperimentState:
            for listener in experiment.listeners:
                state = listener.before_start(state)

            return state

        @override
        def _conduct_experiment(self, experiment: Experiment, state: ExperimentState) -> ExperimentState:
            listeners = experiment.listeners

            for split in experiment.dataset_splitter.split(state):
                training_state = split.get_state(DatasetType.TRAINING)

                if training_state:
                    for listener in listeners:
                        training_state = listener.on_start(training_state)

                    for listener in listeners:
                        training_state = listener.before_training(training_state)

                    for dataset_type in [DatasetType.TRAINING, DatasetType.TEST]:
                        prediction_state = replace(training_state, dataset_type=dataset_type)

                        for listener in listeners:
                            listener.after_prediction(prediction_state)

                    for listener in listeners:
                        training_state = listener.after_training(training_state)

            return state

        """
        The procedure that is used to write evaluation results that have been aggregated across several experiments to
        one or several sinks.
        """

        def __init__(self, evaluation_by_dataset_type: Dict[DatasetType, Dict[str, Table]]):
            """
            :param evaluation_by_dataset_type: A dictionary that stores aggregated evaluation results for different
                                               datasets, mapped to a dataset type
            """
            self.evaluation_by_dataset_type = evaluation_by_dataset_type

        @override
        def _before_experiment(self, _: Experiment, state: ExperimentState) -> ExperimentState:
            input_data = InputData(properties=AggregatedEvaluationResult.PROPERTIES,
                                   context=AggregatedEvaluationResult.CONTEXT)

            for dataset_type, evaluation_by_dataset in self.evaluation_by_dataset_type.items():
                new_state = replace(state, dataset_type=dataset_type)
                input_data_key = input_data.get_key(new_state)
                state.extras[input_data_key] = AggregatedEvaluationResult(evaluation_by_dataset)

            return state

        @override
        def _conduct_experiment(self, experiment: Experiment, state: ExperimentState) -> ExperimentState:
            listeners = experiment.listeners

            for dataset_type in self.evaluation_by_dataset_type.keys():
                prediction_state = replace(state, dataset_type=dataset_type)

                for listener in listeners:
                    listener.after_prediction(prediction_state)

            return state

    @staticmethod
    def __get_batch(arguments: Set[Argument], args: Namespace, meta_data: MetaData) -> List[Command]:
        main_command = meta_data.command
        child_commands = meta_data.child_commands

        if child_commands:
            unique_commands: Set[Command] = set()
            batch = []

            for command in child_commands:
                if BatchMode.SEPARATE_FOLDS.get_value(main_command.apply_to_namespace(args)):
                    command = ReadMode.__remove_fold_option(arguments, args, command)

                if command not in unique_commands:
                    unique_commands.add(command)
                    batch.append(command)

            return batch

        return [main_command]

    @staticmethod
    def __remove_fold_option(arguments: Set[Argument], args: Namespace, command: Command) -> Command:
        command_args = ReadMode.__create_command_args(arguments, args, command)
        value, options = DatasetSplitterArguments.DATASET_SPLITTER.get_value_and_options(command_args)

        if value == DatasetSplitterArguments.VALUE_CROSS_VALIDATION:
            first_fold = options.get_int(DatasetSplitterArguments.OPTION_FIRST_FOLD, default_value=0)
            last_fold = options.get_int(DatasetSplitterArguments.OPTION_LAST_FOLD, default_value=0)

            if first_fold > 0 and first_fold == last_fold:
                options = Options({
                    key: value
                    for key, value in options.dictionary.items() if key not in
                    {DatasetSplitterArguments.OPTION_FIRST_FOLD, DatasetSplitterArguments.OPTION_LAST_FOLD}
                })
                return Command.with_argument(command, DatasetSplitterArguments.DATASET_SPLITTER.name,
                                             value + str(options))

        return command

    @staticmethod
    def __create_command_args(arguments: Set[Argument], args: Namespace, command: Command) -> Namespace:
        ignored_arguments = set(argument_name for argument_names in map(lambda arg: arg.names, arguments)
                                for argument_name in argument_names)
        return command.apply_to_namespace(args,
                                          ignore=ignored_arguments | GroundTruthArguments.PRINT_GROUND_TRUTH.names
                                          | GroundTruthArguments.SAVE_GROUND_TRUTH.names)

    @staticmethod
    def __group_batch_by_dataset(arguments: Set[Argument], args: Namespace,
                                 batch: List[Command]) -> Dict[str, List[Tuple[Command, Namespace]]]:
        commands_by_dataset: Dict[str, List[Tuple[Command, Namespace]]] = {}

        for command in batch:
            command_args = ReadMode.__create_command_args(arguments, args, command)
            dataset_name = DatasetArguments.DATASET_NAME.get_value(command_args)
            commands = commands_by_dataset.setdefault(dataset_name, [])
            commands.append((command, command_args))

        return commands_by_dataset

    def __run_single_experiment(self, args: Namespace, recipe: Recipe, input_directory: Path,
                                command: Command) -> ExperimentState:
        log.info('The command "%s" has been used originally for running this experiment', str(command))
        experiment_builder = recipe.create_experiment_builder(experiment_mode=self.to_enum(),
                                                              args=args,
                                                              command=command,
                                                              load_dataset=False)

        for output_writer in sorted(experiment_builder.output_writers, key=str):
            input_reader = output_writer.create_input_reader(command.to_namespace(), input_directory)

            if input_reader and input_reader.sources:
                experiment_builder.add_input_readers(input_reader)

        experiment = experiment_builder.build(args)
        return ReadMode.SingleExperimentProcedure().conduct_experiment(experiment)

    @staticmethod
    def __aggregate_evaluation(commands_and_their_states: List[Tuple[Command, ExperimentState]],
                               algorithmic_arguments: Set[Argument], dataset_name: str,
                               dataset_type: DatasetType) -> Optional[Table]:
        num_commands = len(commands_and_their_states)

        if num_commands > 1:
            input_data = TabularInputData(properties=EvaluationResult.PROPERTIES, context=EvaluationResult.CONTEXT)
            algorithmic_argument_names = set(map(lambda arg: arg.name, algorithmic_arguments))
            tables: List[Table] = []
            headers: Set[str] = set()

            for command, result_state in commands_and_their_states:
                input_data_key = input_data.get_key(replace(result_state, dataset_type=dataset_type))
                extra = result_state.extras.get(input_data_key)

                if isinstance(extra, Table):
                    tables.append(extra)

                    for argument in command.argument_dict.keys():
                        if argument in algorithmic_argument_names:
                            headers.add(f'{AggregatedEvaluationResult.COLUMN_PREFIX_PARAMETER} {argument}')

            num_tables = len(tables)
            num_missing = num_commands - num_tables

            if num_missing > 0:
                if num_tables > 0:
                    log.error('Evaluation results for %s data of the dataset "%s" are incomplete. %s of %s %s missing.',
                              dataset_type, dataset_name, num_missing, num_commands,
                              'files are' if num_missing > 1 else 'file is')
            else:
                return ReadMode.__aggregate_tables(commands_and_their_states, headers, tables)

        return None

    @staticmethod
    def __aggregate_tables(commands_and_their_states: List[Tuple[Command, ExperimentState]], headers: Set[str],
                           tables: List[Table]) -> Table:
        aggregated_table = RowWiseTable.aggregate(*tables).to_column_wise_table()

        for position, header in enumerate(sorted(headers)):
            column: List[Cell] = []
            unique_values: Set[Cell] = set()

            for command, _ in commands_and_their_states:
                argument_name = header[len(AggregatedEvaluationResult.COLUMN_PREFIX_PARAMETER):].lstrip()
                argument_value = command.argument_dict.get(argument_name)
                column.append(argument_value)
                unique_values.add(argument_value)

            if len(unique_values) > 1:
                aggregated_table.add_column(*column, header=header, position=position)

        return aggregated_table

    def __write_aggregated_evaluation_result(self, args: Namespace, recipe: Recipe, command: Command,
                                             evaluation_by_dataset_type: Dict[DatasetType, Dict[str, Table]]):
        experiment_builder = recipe.create_experiment_builder(experiment_mode=self.to_enum(),
                                                              args=args,
                                                              command=command,
                                                              load_dataset=False)

        experiment = experiment_builder.build(args)
        return ReadMode.AggregateEvaluationProcedure(evaluation_by_dataset_type).conduct_experiment(experiment)

    @override
    def _run_experiment(self, extension_arguments: Set[Argument], algorithmic_arguments: Set[Argument], args: Namespace,
                        recipe: Recipe, meta_data: MetaData, input_directory: Path):
        batch = self.__get_batch(extension_arguments, args, meta_data)
        num_experiments = len(batch)
        log.info('Reading experimental results of %s %s...', num_experiments,
                 'experiments' if num_experiments > 1 else 'experiment')
        i = 1

        evaluation_by_dataset_type: Dict[DatasetType, Dict[str, Table]] = {}

        for dataset_name, commands in self.__group_batch_by_dataset(extension_arguments, args, batch).items():
            commands_and_their_states: List[Tuple[Command, ExperimentState]] = []

            for command, command_args in commands:
                log.info('\nReading experimental results of experiment (%s / %s)...', i, num_experiments)
                state = self.__run_single_experiment(command_args, recipe, input_directory, command)
                commands_and_their_states.append((command, state))
                i += 1

            for dataset_type in [DatasetType.TRAINING, DatasetType.TEST]:
                table = self.__aggregate_evaluation(commands_and_their_states,
                                                    algorithmic_arguments,
                                                    dataset_name=dataset_name,
                                                    dataset_type=dataset_type)

                if table:
                    evaluation_by_dataset = evaluation_by_dataset_type.setdefault(dataset_type, {})
                    evaluation_by_dataset[dataset_name] = table

        self.__write_aggregated_evaluation_result(args, recipe, meta_data.command, evaluation_by_dataset_type)

    @override
    def to_enum(self) -> ExperimentMode:
        return ExperimentMode.READ
