"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility classes to be used by different modes of operation.
"""
import logging as log

from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from typing import override

from mlrl.testbed.command import Command
from mlrl.testbed.experiments.dataset_type import DatasetType
from mlrl.testbed.experiments.experiment import Experiment, ExperimentalProcedure
from mlrl.testbed.experiments.input.policies import MissingInputPolicy
from mlrl.testbed.experiments.output.sinks.sink import FileSink
from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.experiments.state import ExperimentMode, ExperimentState


class OutputUtil:
    """
    A utility class for reading experimental results produced by a specific command.
    """

    class ReadProcedure(ExperimentalProcedure):
        """
        A procedure that allows reading the output files produced by a single experiment.
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

    def __init__(self,
                 args: Namespace,
                 recipe: Recipe,
                 command: Command,
                 input_directory: Path,
                 file_sinks_only: bool = False):
        """
        :param args:            The command line arguments provided by the user
        :param recipe:          A `Recipe` that provides access to the ingredients that are needed for setting up
                                experiments
        :param command:         The command that produced the experimental results to be read
        :param input_directory: The path to the directory from which the experimental results should be read
        :param file_sinks_only: True, if sources should only be considered if they result from a `FileSink`, False
                                otherwise
        """
        experiment_builder = recipe.create_experiment_builder(experiment_mode=ExperimentMode.READ,
                                                              args=args,
                                                              command=command,
                                                              load_dataset=False)
        command_args = command.to_namespace()

        for output_writer in sorted(experiment_builder.output_writers, key=str):
            if not file_sinks_only or any(isinstance(sink, FileSink) for sink in output_writer.sinks):
                input_reader = output_writer.create_input_reader(command_args, input_directory)

                if input_reader and input_reader.sources:
                    experiment_builder.add_input_readers(input_reader)

        self.args = args
        self.experiment_builder = experiment_builder

    def read_output_files(self) -> ExperimentState:
        """
        Reads the experimental results and writes them to sinks.

        :return: A state that stores the experimental results that have been read in its extras
        """
        experiment = self.experiment_builder.build(self.args)
        return OutputUtil.ReadProcedure().conduct_experiment(experiment)

    def check_if_output_files_exist(self) -> bool:
        """
        Returns whether all experimental results do already exist or not.

        :return: True, if all experimental results do already exist, False otherwise
        """
        try:
            log.disable(log.CRITICAL)  # Temporarily disable logging
            experiment_builder = self.experiment_builder

            if experiment_builder.input_readers:
                for output_writer in experiment_builder.output_writers:
                    output_writer.sinks.clear()

                experiment_builder.set_missing_input_policy(MissingInputPolicy.EXIT)
                experiment = experiment_builder.build(self.args)
                OutputUtil.ReadProcedure().conduct_experiment(experiment)
                return True
        except IOError:
            return False
        finally:
            log.disable(log.NOTSET)  # Re-enable logging

        return False
