"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing meta-data that is part of input data.
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, override

from mlrl.testbed.command import Command
from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.input.data import StructuralInputData
from mlrl.testbed.experiments.meta_data import MetaData
from mlrl.testbed.experiments.output.meta_data.meta_data import OutputMetaData
from mlrl.testbed.experiments.state import ExperimentState

from mlrl.util.version import Version


class InputMetaData(StructuralInputData):
    """
    Represents meta-data that is part of input data.
    """

    SCHEMA_FILE_PATH = Path(__file__).parent / 'metadata.schema.yml'

    def __init__(self):
        super().__init__(StructuralInputData.Properties(file_name=OutputMetaData.FILENAME),
                         Context(include_dataset_type=False, include_prediction_scope=False))

    @override
    def _update_state(self, state: ExperimentState, dictionary: Dict[Any, Any]):
        version = Version.parse(dictionary[OutputMetaData.ATTRIBUTE_VERSION], skip_on_error=True)
        timestamp = datetime.strptime(dictionary[OutputMetaData.ATTRIBUTE_TIMESTAMP], MetaData.TIMESTAMP_FORMAT)
        command = Command.from_string(dictionary[OutputMetaData.ATTRIBUTE_COMMAND])
        meta_data = MetaData(command=command, version=version, timestamp=timestamp)

        if OutputMetaData.ATTRIBUTE_CHILD_COMMANDS in dictionary:
            meta_data.child_commands = [
                Command.from_string(child_command)
                for child_command in dictionary[OutputMetaData.ATTRIBUTE_CHILD_COMMANDS]
            ]

        state.meta_data = meta_data
