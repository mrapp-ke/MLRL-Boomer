"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement a mode of operation for reading experimental results.
"""
import logging as log

from argparse import Namespace
from pathlib import Path
from typing import override

from mlrl.testbed.experiments.meta_data import MetaData
from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.modes.mode import InputMode


class ReadMode(InputMode):
    """
    An abstract base class for all modes of operation that read experimental results.
    """

    @override
    def _run_experiment(self, args: Namespace, recipe: Recipe, meta_data: MetaData, input_directory: Path):
        batch = meta_data.child_commands if meta_data.child_commands else [meta_data.command]
        num_experiments = len(batch)

        log.info('Reading experimental results of %s %s...', num_experiments,
                 'experiments' if num_experiments > 1 else 'experiment')

        for i, command in enumerate(batch):
            log.info('\nReading experimental results of experiment (%s / %s): "%s"', i + 1, num_experiments,
                     str(command))
            experiment_builder = recipe.create_experiment_builder(command.apply_to_namespace(args), command)

            for output_writer in experiment_builder.output_writers:
                input_reader = output_writer.create_input_reader(command.apply_to_namespace(Namespace()),
                                                                 input_directory)

                if input_reader and input_reader.sources:
                    experiment_builder.add_input_readers(input_reader)

            experiment_builder.run(train_model=False, measure_runtime=False, enable_logging=False)
