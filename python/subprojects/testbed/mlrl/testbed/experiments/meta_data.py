"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing meta-data.
"""
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import version
from typing import List

from mlrl.testbed.command import Command

from mlrl.util.version import Version


@dataclass
class MetaData:
    """
    Meta-data that provides information about a command that has been used for running MLRL-Testbed.

    Attributes:
        command:        The command
        child_commands: Commands run by the original command, e.g., in batch mode
        version:        The version of MLRL-Testbed used for running the command
        timestamp:      The date and time when the command was run
    """
    command: Command = field(default_factory=Command.from_argv)
    child_commands: List[Command] = field(default_factory=list)
    version: Version = field(default_factory=lambda: Version.parse(version('mlrl-testbed'), skip_on_error=True))
    timestamp: datetime = datetime.now()

    TIMESTAMP_FORMAT = '%Y-%m-%d_%H-%M'

    @property
    def formatted_timestamp(self) -> str:
        """
        The timestamp in a human-readable format.
        """
        return self.timestamp.strftime(self.TIMESTAMP_FORMAT)
