"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow reading input data from YAML files.
"""

from pathlib import Path
from typing import Any, Dict, Optional, override

from mlrl.testbed.experiments.input.data import StructuralInputData
from mlrl.testbed.experiments.input.sources.source import StructuralFileSource
from mlrl.testbed.util.yml import read_and_validate_yaml, read_yaml


class YamlFileSource(StructuralFileSource):
    """
    Allows to read structural input data from a YAML file.
    """

    SUFFIX_YAML = 'yml'

    def __init__(self, directory: Path, schema_file_path: Optional[Path] = None):
        """
        :param directory:           The path to the directory of the file
        :param schema_file_path:    An optional path to a YAML schema file
        """
        super().__init__(directory=directory, suffix=self.SUFFIX_YAML)
        self.schema_file_path = schema_file_path

    @override
    def _read_dictionary_from_file(self, file_path: Path, _: StructuralInputData) -> Optional[Dict[Any, Any]]:
        schema_file_path = self.schema_file_path

        if schema_file_path:
            return read_and_validate_yaml(yaml_file_path=file_path, schema_file_path=schema_file_path)

        return read_yaml(yaml_file_path=file_path)
