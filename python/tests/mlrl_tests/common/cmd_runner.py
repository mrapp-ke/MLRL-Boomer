"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
import shutil
import subprocess

from functools import reduce
from os import environ
from pathlib import Path
from typing import List, Optional

import pytest

from .cmd_builder import CmdBuilder
from .comparison import FileComparison, TextFileComparison

from mlrl.testbed.experiments.state import ExperimentMode


class CmdRunner:
    """
    Allows to run commands that have been configured via a `CmdBuilder`.
    """

    def __format_cmd(self, args: Optional[List[str]] = None) -> str:
        args = self.args if args is None else args
        return reduce(lambda aggr, arg: aggr + (' ' + arg if len(aggr) > 0 else arg), args, '')

    def __run_cmd(self):
        args = self.args
        out = subprocess.run(args, capture_output=True, text=True, check=False)
        exit_code = out.returncode

        if exit_code != 0:
            pytest.fail('Command "' + self.__format_cmd(args) + '" terminated with non-zero exit code ' + str(exit_code)
                        + '\n\n' + str(out.stderr))

        return out

    def __assert_file_exists(self, file: Path):
        if not file.is_file():
            pytest.fail('Command "' + self.__format_cmd() + '" is expected to create file "' + str(file)
                        + '", but it does not exist')

    def __compare_or_overwrite_output_files(self, output_dir: Path, expected_output_dir: Path):
        expected_files_to_be_deleted = []

        for expected_file in expected_output_dir.rglob('*'):
            if expected_file.is_file() and expected_file != expected_output_dir / 'std.out' and \
                    not any (parent.name == CmdBuilder.RERUN_DIR.name for parent in expected_file.parents):
                actual_file = output_dir / expected_file.relative_to(expected_output_dir)

                if self.__should_overwrite_files():
                    if not actual_file.is_file():
                        expected_files_to_be_deleted.append(expected_file)
                else:
                    self.__assert_file_exists(actual_file)

        for expected_file in expected_files_to_be_deleted:
            expected_file.unlink()

        # Check if all output files have the expected content...
        if output_dir.is_dir():
            for actual_file in output_dir.rglob('*'):
                if actual_file.is_file():
                    expected_file = expected_output_dir / actual_file.relative_to(output_dir)
                    self.__compare_or_overwrite_files(FileComparison.for_file(actual_file), expected_file=expected_file)

    def __compare_or_overwrite_files(self, file_comparison: FileComparison, expected_file: Path):
        overwrite = self.__should_overwrite_files()
        difference = file_comparison.compare_or_overwrite(expected_file, overwrite=overwrite)

        if difference:
            pytest.fail('Command "' + self.__format_cmd() + '" resulted in unexpected output: ' + str(difference))

    @staticmethod
    def __should_overwrite_files() -> bool:
        variable_name = 'OVERWRITE_OUTPUT_FILES'
        value = environ.get(variable_name, 'false').strip().lower()

        if value == 'true':
            return True
        if value == 'false':
            return False
        raise ValueError('Value of environment variable "' + variable_name + '" must be "true" or "false", but is "'
                         + value + '"')

    def __init__(self, builder: CmdBuilder):
        """
        :param builder: A `CmdBuilder` that has been used for configuring a command
        """
        self.builder = builder
        self.args = builder.build()

    def run(self, test_name: str, wipe_before: bool = True, wipe_after: bool = True, compare_output: bool = True):
        """
        Runs the command.

        :param test_name:       The name of the directory that stores the output files produced by the command
        :param wipe_before:     True, if temporary directory should be deleted before running the command, False
                                otherwise
        :param wipe_after:      True, if temporary directory should be deleted after running the command, False
                                otherwise
        :param compare_output:  True, if the output of the command should be compared to the expected output, False
                                otherwise
        """
        builder = self.builder
        output_dir = builder.base_dir

        # Delete temporary directories...
        if wipe_before:
            shutil.rmtree(output_dir, ignore_errors=True)

        # Run command...
        out = self.__run_cmd()

        if builder.model_save_dir and builder.model_load_dir:
            shutil.rmtree(output_dir, ignore_errors=True)
            out = self.__run_cmd()

        # Check if output of the command is as expected...
        if compare_output:
            stdout = [self.__format_cmd()] + str(out.stdout).splitlines()
            actual_output_dir = output_dir
            expected_output_dir = builder.expected_output_dir / test_name

            if builder.mode in {ExperimentMode.RUN, ExperimentMode.READ} and not builder.show_help:
                expected_output_dir = expected_output_dir / CmdBuilder.RERUN_DIR
                actual_output_dir = actual_output_dir / CmdBuilder.RERUN_DIR

            self.__compare_or_overwrite_files(TextFileComparison(stdout), expected_file=expected_output_dir / 'std.out')

            # Check if all expected files have been created...
            self.__compare_or_overwrite_output_files(output_dir=actual_output_dir,
                                                     expected_output_dir=expected_output_dir)

        # Delete temporary directories...
        if wipe_after:
            shutil.rmtree(output_dir, ignore_errors=True)
