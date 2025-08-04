"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write meta-data to one or several sinks.
"""
from argparse import Namespace
from pathlib import Path
from typing import Set, override

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.sinks import YamlFileSink
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, BoolArgument


class MetaDataExtension(Extension):
    """
    An extension that configures the functionality to write meta-data to one or several sinks.
    """

    SAVE_META_DATA = BoolArgument(
        '--save-meta-data',
        default=False,
        description='Whether meta-data should be saved to output files or not.',
    )

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
        return {self.SAVE_META_DATA}

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        if self.SAVE_META_DATA.get_value(args):
            base_dir = OutputExtension.BASE_DIR.get_value(args)

            if base_dir:
                create_directory = OutputExtension.CREATE_DIRS.get_value(args)
                experiment_builder.meta_data_writer.add_sinks(
                    YamlFileSink(directory=Path(base_dir), create_directory=create_directory))
