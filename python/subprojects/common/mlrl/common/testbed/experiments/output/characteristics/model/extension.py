"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write characteristics of rule models to one or several
sinks.
"""
from argparse import Namespace
from pathlib import Path
from typing import List, Set, override

from mlrl.common.testbed.experiments.output.characteristics.model.writer import RuleModelCharacteristicsWriter

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.arguments import OutputArguments, ResultDirectoryArguments
from mlrl.testbed.experiments.output.extension import OutputExtension, ResultDirectoryExtension
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, BoolArgument


class RuleModelCharacteristicsExtension(Extension):
    """
    An extension that configures the functionality to write characteristics of rule models to one or several sinks.
    """

    PRINT_MODEL_CHARACTERISTICS = BoolArgument(
        '--print-model-characteristics',
        description='Whether the characteristics of models should be printed on the console or not.',
    )

    SAVE_MODEL_CHARACTERISTICS = BoolArgument(
        '--save-model-characteristics',
        description='Whether the characteristics of models should be written to output files or not.',
    )

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(OutputExtension(), ResultDirectoryExtension(), *dependencies)

    @override
    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.PRINT_MODEL_CHARACTERISTICS, self.SAVE_MODEL_CHARACTERISTICS}

    def __create_log_sinks(self, args: Namespace) -> List[Sink]:
        if self.PRINT_MODEL_CHARACTERISTICS.get_value(args, default=OutputArguments.PRINT_ALL.get_value(args)):
            return [LogSink()]
        return []

    def __create_csv_file_sinks(self, args: Namespace) -> List[Sink]:
        value = self.SAVE_MODEL_CHARACTERISTICS.get_value(args, default=OutputArguments.SAVE_ALL.get_value(args))
        result_directory = ResultDirectoryArguments.RESULT_DIR.get_value(args)

        if value and result_directory:
            return [
                CsvFileSink(directory=Path(result_directory),
                            create_directory=OutputArguments.CREATE_DIRS.get_value(args))
            ]
        return []

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_csv_file_sinks(args)

        if sinks:
            writer = RuleModelCharacteristicsWriter().add_sinks(*sinks)
            experiment_builder.add_post_training_output_writers(writer)
