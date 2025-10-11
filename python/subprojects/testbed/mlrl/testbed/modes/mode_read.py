"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement a mode of operation for reading experimental results.
"""
import logging as log

from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from typing import override

from mlrl.testbed_sklearn.experiments.output.dataset.arguments_ground_truth import GroundTruthArguments

from mlrl.testbed.experiments.dataset_type import DatasetType
from mlrl.testbed.experiments.experiment import Experiment, ExperimentalProcedure
from mlrl.testbed.experiments.meta_data import MetaData
from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.experiments.state import ExperimentMode, ExperimentState
from mlrl.testbed.modes.mode import InputMode

from mlrl.util.cli import Argument


class ReadMode(InputMode):
    """
    An abstract base class for all modes of operation that read experimental results.
    """

    class Procedure(ExperimentalProcedure):
        """
        The procedure that is used to conduct experiments in read mode.
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

    def __init__(self, *read_mode_arguments: Argument):
        """
        :param read_mode_arguments: The arguments that can be specified by the user in read mode
        """
        self._read_mode_arguments = set(argument_name
                                        for argument_names in map(lambda arg: arg.names, read_mode_arguments)
                                        for argument_name in argument_names)

    @override
    def _run_experiment(self, args: Namespace, recipe: Recipe, meta_data: MetaData, input_directory: Path):
        batch = meta_data.child_commands if meta_data.child_commands else [meta_data.command]
        num_experiments = len(batch)

        log.info('Reading experimental results of %s %s...', num_experiments,
                 'experiments' if num_experiments > 1 else 'experiment')

        for i, command in enumerate(batch):
            log.info('\nReading experimental results of experiment (%s / %s)...', i + 1, num_experiments)
            log.info('The command "%s" has been used originally for running this experiment', str(command))
            command_args = command.apply_to_namespace(args,
                                                      ignore=self._read_mode_arguments
                                                      | GroundTruthArguments.PRINT_GROUND_TRUTH.names
                                                      | GroundTruthArguments.SAVE_GROUND_TRUTH.names)
            experiment_builder = recipe.create_experiment_builder(experiment_mode=self.to_enum(),
                                                                  args=command_args,
                                                                  command=command,
                                                                  load_dataset=False)

            for output_writer in sorted(experiment_builder.output_writers, key=str):
                input_reader = output_writer.create_input_reader(command.apply_to_namespace(Namespace()),
                                                                 input_directory)

                if input_reader and input_reader.sources:
                    experiment_builder.add_input_readers(input_reader)

            ReadMode.Procedure().conduct_experiment(experiment_builder.build(command_args))

    @override
    def to_enum(self) -> ExperimentMode:
        return ExperimentMode.READ
