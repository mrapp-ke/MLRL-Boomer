"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
import shutil
import subprocess

from functools import reduce
from os import environ, listdir, makedirs, path, remove
from typing import Any, Optional

from .cmd_builder import CmdBuilder
from .comparison import FileComparison, TextFileComparison


class CmdRunner:
    """
    Allows to run commands that have been configured via a `CmdBuilder`.
    """

    def __create_temporary_directories(self):
        builder = self.builder
        makedirs(builder.output_dir, exist_ok=True)
        model_dir = builder.model_dir

        if model_dir:
            makedirs(model_dir, exist_ok=True)

    def __delete_temporary_directories(self):
        builder = self.builder
        shutil.rmtree(builder.output_dir, ignore_errors=True)
        model_dir = builder.model_dir

        if model_dir:
            shutil.rmtree(model_dir, ignore_errors=True)

    def __format_cmd(self):
        return reduce(lambda aggr, arg: aggr + (' ' + arg if len(aggr) > 0 else arg), self.args, '')

    def __run_cmd(self):
        out = subprocess.run(self.args, capture_output=True, text=True, check=False)
        exit_code = out.returncode

        if exit_code != 0:
            self.test_case.fail('Command "' + self.__format_cmd() + '" terminated with non-zero exit code '
                                + str(exit_code) + '\n\n' + str(out.stderr))

        return out

    def __assert_model_files_exist(self):
        builder = self.builder
        self.__assert_files_exist(directory=builder.model_dir, file_name='model', suffix='pickle')

    def __assert_files_exist(self, directory: Optional[str], file_name: str, suffix: str):
        if directory:
            builder = self.builder
            num_folds = builder.num_folds

            if num_folds > 0:
                current_fold = builder.current_fold

                if current_fold > 0:
                    self.__assert_file_exists(directory, self.__get_file_name(file_name, suffix, current_fold))
                else:
                    for i in range(num_folds):
                        self.__assert_file_exists(directory, self.__get_file_name(file_name, suffix, i + 1))
            else:
                self.__assert_file_exists(directory, self.__get_file_name(file_name, suffix))

    @staticmethod
    def __get_file_name(name: str, suffix: str, fold: Optional[int] = None):
        if fold is None:
            return name + '.' + suffix
        return name + '_fold-' + str(fold) + '.' + suffix

    def __assert_file_exists(self, directory: str, file_name: str):
        """
        Asserts that a specific file exists.

        :param directory:   The path to the directory where the file should be located
        :param file_name:   The name of the file
        """
        file = path.join(directory, file_name)

        if not path.isfile(file):
            self.test_case.fail('Command "' + self.__format_cmd() + '" is expected to create file ' + str(file)
                                + ', but it does not exist')

    def __compare_or_overwrite_files(self, file_comparison: FileComparison, test_name: str, file_name: str):
        expected_output_file = path.join(self.builder.expected_output_dir, test_name, file_name)
        overwrite = self.__should_overwrite_files()
        difference = file_comparison.compare_or_overwrite(expected_output_file, overwrite=overwrite)

        if difference:
            self.test_case.fail('Command "' + self.__format_cmd() + '" resulted in unexpected output: '
                                + str(difference))

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

    def __init__(self, test_case: Any, builder: CmdBuilder):
        """
        :param test_case:   The test case from which the command should be run
        :param builder:     A `CmdBuilder` that has been used for configuring a command
        """
        self.test_case = test_case
        self.builder = builder
        self.args = builder.build()

    def run(self, test_name: str):
        """
        Runs the command.

        :param test_name: The name of the directory that stores the output files produced by the command
        """
        builder = self.builder

        # Create temporary directories...
        self.__create_temporary_directories()

        # Run command...
        out = self.__run_cmd()

        if builder.model_dir:
            out = self.__run_cmd()

        # Check if output of the command is as expected...
        stdout = [self.__format_cmd()] + str(out.stdout).splitlines()
        stdout_file = 'std.out'
        self.__compare_or_overwrite_files(TextFileComparison(stdout), test_name=test_name, file_name=stdout_file)

        # Check if all expected output files have been created...
        self.__assert_model_files_exist()
        output_dir = builder.output_dir
        expected_output_dir = builder.expected_output_dir
        expected_files_to_be_deleted = []

        for expected_file in listdir(path.join(expected_output_dir, test_name)):
            if expected_file != stdout_file:
                if self.__should_overwrite_files():
                    if not path.isfile(path.join(output_dir, expected_file)):
                        expected_files_to_be_deleted.append(path.join(expected_output_dir, test_name, expected_file))
                else:
                    self.__assert_file_exists(output_dir, expected_file)

        for expected_file in expected_files_to_be_deleted:
            remove(expected_file)

        # Check if all output files have the expected content...
        for output_file in listdir(output_dir):
            self.__compare_or_overwrite_files(FileComparison.for_file(path.join(output_dir, output_file)),
                                              test_name=test_name,
                                              file_name=path.basename(output_file))

        # Delete temporary directories...
        self.__delete_temporary_directories()
