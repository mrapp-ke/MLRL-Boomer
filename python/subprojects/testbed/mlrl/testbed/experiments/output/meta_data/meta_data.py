"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing meta-data that is part of output data.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from mlrl.testbed.command import Command


@dataclass
class MetaData:
    """
    Meta-data that provides information about a command that has been used for running MLRL-Testbed.

    Attributes:
        version:        The version of MLRL-Testbed used for running the command
        timestamp:      The date and time when the command was run
        command:        The command
        child_commands: Commands run by the original command, e.g., in batch mode
    """
    version: str
    timestamp: datetime
    command: Command
    child_commands: List[Command] = field(default_factory=list)

    @property
    def formatted_timestamp(self) -> str:
        """
        The timestamp in a human-readable format.
        """
        return self.timestamp.strftime('%Y-%m-%d_%H-%M')
