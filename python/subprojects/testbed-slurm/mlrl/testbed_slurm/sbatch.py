"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run "sbatch" commands.
"""
import subprocess

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, override


class SlurmCommand(ABC):
    """
    An abstract base class for all Slurm commands.
    """

    @dataclass
    class Result:
        """
        The result of a Slurm command.

        Attributes:
            exit_code:  The exit code of the command
            output:     The stdout or stderr output of the command
        """
        exit_code: int
        output: str

        @property
        def ok(self) -> bool:
            """
            True, if the command terminated with a zero exit code, False otherwise.
            """
            return self.exit_code == 0

    def __init__(self, command: str):
        """
        :param command: The command to be run
        """
        self.command = command

    def run(self) -> Result:
        """
        Runs the command.

        :return: The result of the command
        """
        command = [self.command] + self._get_arguments()
        output = subprocess.run(command, check=False, text=True, capture_output=True)
        exit_code = output.returncode
        output_str = str(output.stdout if exit_code == 0 else output.stderr).strip()
        return SlurmCommand.Result(exit_code=exit_code, output=output_str)

    @abstractmethod
    def _get_arguments(self) -> List[str]:
        """
        Must be implemented by subclasses in order to return the arguments of the command to be run.

        :return: A list that contains the arguments
        """


class Sbatch(SlurmCommand):
    """
    Allows to run "sbatch" commands.
    """

    def __init__(self):
        super().__init__('sbatch')
        self.flag_version = False
        self.arguments = []

    def version(self) -> 'Sbatch':
        """
        Sets the "--version" flag.

        :return: The object itself
        """
        self.flag_version = True
        return self

    def script(self, path: Path) -> 'Sbatch':
        """
        Sets an absolute or relative path to a batch script that defines the job to be run.
        """
        self.arguments.append(str(path))
        return self

    @override
    def _get_arguments(self) -> List[str]:
        return ['--version'] if self.flag_version else self.arguments
