"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing meta-data that is part of output data.
"""
from typing import Any, Dict, Optional, override

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.meta_data import MetaData
from mlrl.testbed.experiments.output.data import OutputData, StructuralOutputData

from mlrl.util.options import Options


class OutputMetaData(StructuralOutputData):
    """
    Represents meta-data that is part of output data.
    """

    ATTRIBUTE_VERSION = 'version'

    ATTRIBUTE_TIMESTAMP = 'timestamp'

    ATTRIBUTE_COMMAND = 'command'

    ATTRIBUTE_CHILD_COMMANDS = 'child-commands'

    def __init__(self, meta_data: MetaData):
        """
        :param meta_data: The meta-data
        """
        super().__init__(OutputData.Properties(name='Meta data', file_name='metadata'),
                         Context(include_dataset_type=False, include_prediction_scope=False))
        self.meta_data = meta_data

    @override
    def to_dict(self, options: Options, **kwargs) -> Optional[Dict[Any, Any]]:
        """
        See :func:`mlrl.testbed.experiments.output.data.StructuralOutputData.to_dict`
        """
        meta_data = self.meta_data
        dictionary: Dict[Any, Any] = {
            self.ATTRIBUTE_VERSION: meta_data.version,
            self.ATTRIBUTE_TIMESTAMP: meta_data.formatted_timestamp,
            self.ATTRIBUTE_COMMAND: str(meta_data.command),
        }

        if meta_data.child_commands:
            dictionary[self.ATTRIBUTE_CHILD_COMMANDS] = [str(command) for command in meta_data.child_commands]

        return dictionary
