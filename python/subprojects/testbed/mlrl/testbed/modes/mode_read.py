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
from mlrl.testbed.experiments.input.data import TabularInputData
from mlrl.testbed.experiments.input.dataset.arguments import DatasetArguments
from mlrl.testbed.experiments.input.dataset.splitters.arguments import DatasetSplitterArguments
from mlrl.testbed.experiments.meta_data import MetaData
from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.experiments.state import ExperimentMode, ExperimentState
from mlrl.testbed.experiments.table import RowWiseTable, Table
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

    @staticmethod
    def __get_batch(arguments: List[Argument], args: Namespace, meta_data: MetaData) -> List[Command]:
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
    def __remove_fold_option(arguments: List[Argument], args: Namespace, command: Command) -> Command:
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
    def __create_command_args(arguments: List[Argument], args: Namespace, command: Command) -> Namespace:
        ignored_arguments = set(argument_name for argument_names in map(lambda arg: arg.names, arguments)
                                for argument_name in argument_names)
        return command.apply_to_namespace(args,
                                          ignore=ignored_arguments | GroundTruthArguments.PRINT_GROUND_TRUTH.names
                                          | GroundTruthArguments.SAVE_GROUND_TRUTH.names)

    @staticmethod
    def __group_batch_by_dataset(arguments: List[Argument], args: Namespace,
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
            input_reader = output_writer.create_input_reader(command.apply_to_namespace(Namespace()), input_directory)

            if input_reader and input_reader.sources:
                experiment_builder.add_input_readers(input_reader)

        experiment = experiment_builder.build(args)
        return ReadMode.SingleExperimentProcedure().conduct_experiment(experiment)

    @staticmethod
    def __aggregate_evaluation(states: List[ExperimentState], dataset_name: str,
                               dataset_type: DatasetType) -> Optional[Table]:
        num_states = len(states)

        if num_states > 1:
            input_data = TabularInputData(properties=EvaluationResult.PROPERTIES, context=EvaluationResult.CONTEXT)
            tables: List[Table] = []

            for state in states:
                input_data_key = input_data.get_key(replace(state, dataset_type=dataset_type))
                extra = state.extras.get(input_data_key)

                if isinstance(extra, Table):
                    tables.append(extra)

            num_tables = len(tables)
            num_missing = num_states - num_tables

            if num_missing > 0:
                if num_tables > 0:
                    log.error('Evaluation results for %s data of the dataset "%s" are incomplete. %s of %s %s missing.',
                              dataset_type, dataset_name, num_missing, num_states,
                              'files are' if num_missing > 1 else 'file is')
            else:
                return RowWiseTable.aggregate(*tables)

        return None

    @override
    def _run_experiment(self, arguments: List[Argument], args: Namespace, recipe: Recipe, meta_data: MetaData,
                        input_directory: Path):
        batch = self.__get_batch(arguments, args, meta_data)
        num_experiments = len(batch)
        log.info('Reading experimental results of %s %s...', num_experiments,
                 'experiments' if num_experiments > 1 else 'experiment')
        i = 1

        evaluation_by_dataset: Dict[str, Dict[DatasetType, Table]] = {}

        for dataset_name, commands in self.__group_batch_by_dataset(arguments, args, batch).items():
            states: List[ExperimentState] = []

            for command, command_args in commands:
                log.info('\nReading experimental results of experiment (%s / %s)...', i, num_experiments)
                state = self.__run_single_experiment(command_args, recipe, input_directory, command)
                states.append(state)
                i += 1

            for dataset_type in [DatasetType.TRAINING, DatasetType.TEST]:
                table = self.__aggregate_evaluation(states, dataset_name=dataset_name, dataset_type=dataset_type)

                if table:
                    evaluation_by_dataset_type = evaluation_by_dataset.setdefault(dataset_name, {})
                    evaluation_by_dataset_type[dataset_type] = table

    @override
    def to_enum(self) -> ExperimentMode:
        return ExperimentMode.READ
