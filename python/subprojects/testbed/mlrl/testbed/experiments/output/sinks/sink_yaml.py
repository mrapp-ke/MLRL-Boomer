"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing output data to YAML files.
"""
from pathlib import Path
from typing import override

import yaml

from mlrl.testbed.experiments.output.data import OutputData, StructuralOutputData
from mlrl.testbed.experiments.output.sinks.sink import FileSink
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.io import open_writable_file

from mlrl.util.options import Options


class YamlFileSink(FileSink):
    """
    Allows to write structural output data to a YAML file.
    """

    SUFFIX_YAML = 'yml'

    def __init__(self, directory: Path, options: Options = Options(), create_directory: bool = False):
        """
        :param directory:           The path to the directory of the file
        :param options:             Options to be taken into account
        :param create_directory:    True, if the given directory should be created, if it does not exist, False
                                    otherwise
        """
        super().__init__(directory=directory,
                         suffix=self.SUFFIX_YAML,
                         options=options,
                         create_directory=create_directory)

    @override
    def _write_to_file(self, file_path: Path, state: ExperimentState, output_data: OutputData, **kwargs):
        if isinstance(output_data, StructuralOutputData):
            dictionary = output_data.to_dict(self.options, **kwargs)

            if dictionary:
                with open_writable_file(file_path) as yaml_file:
                    yaml.dump(dictionary, yaml_file)
