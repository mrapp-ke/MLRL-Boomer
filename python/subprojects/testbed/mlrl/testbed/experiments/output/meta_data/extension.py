"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write meta-data to one or several sinks.
"""
from argparse import Namespace
from pathlib import Path
from typing import Set, override

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.arguments import OutputArguments
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.meta_data.arguments import MetaDataArguments
from mlrl.testbed.experiments.output.sinks import YamlFileSink
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import AUTO, Argument
from mlrl.util.options import BooleanOption


class MetaDataExtension(Extension):
    """
    An extension that configures the functionality to write meta-data to one or several sinks.
    """

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(OutputExtension(), *dependencies)

    @override
    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {MetaDataArguments.SAVE_META_DATA}

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        save_meta_data = MetaDataArguments.SAVE_META_DATA.get_value(args)

        if save_meta_data == BooleanOption.TRUE or (save_meta_data == AUTO
                                                    and experiment_builder.has_output_file_writers):
            base_dir = OutputArguments.BASE_DIR.get_value(args)

            if base_dir:
                create_directory = OutputArguments.CREATE_DIRS.get_value(args)
                experiment_builder.meta_data_writer.add_sinks(
                    YamlFileSink(directory=Path(base_dir), create_directory=create_directory))
