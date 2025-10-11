"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement a mode of operation for re-running experiments based on their meta-data.
"""
import logging as log
import subprocess

from argparse import Namespace
from pathlib import Path
from typing import override

from mlrl.testbed.command import ArgumentDict, Command
from mlrl.testbed.experiments.meta_data import MetaData
from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.experiments.state import ExperimentMode
from mlrl.testbed.modes.mode import InputMode, Mode

from mlrl.util.format import format_iterable


class RunMode(InputMode):
    """
    An abstract base class for all modes of operation for re-running experiments based on their meta-data.
    """

    @override
    def _run_experiment(self, args: Namespace, recipe: Recipe, meta_data: MetaData, input_directory: Path):
        meta_data_command = meta_data.command
        log.info('Re-running experiment "%s"...', meta_data_command)
        overridden_arguments = Command.from_argv().argument_list.filter(*self.INPUT_DIR.names, *Mode.MODE.names)

        if overridden_arguments:
            meta_data_arguments = meta_data_command.argument_dict
            merged_arguments = meta_data_arguments | overridden_arguments.to_dict()
            meta_data_command = Command.from_dict(meta_data_command.module_name, ArgumentDict(merged_arguments))
            log.info(
                'Arguments "%s" modifying the original experiments have been provided. The resulting command is '
                + '"%s"...', format_iterable(overridden_arguments, separator=' '), meta_data_command)

        log.info('')
        subprocess.run(list(meta_data_command), check=False)

    @override
    def to_enum(self) -> ExperimentMode:
        return ExperimentMode.RUN
