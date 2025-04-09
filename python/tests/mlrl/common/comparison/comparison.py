"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
import re as regex

from abc import ABC
from dataclasses import dataclass
from os import environ, makedirs, path
from typing import Iterable, Optional

from mlrl.testbed.io import ENCODING_UTF8


class Difference(ABC):
    """
    A difference between two files.
    """


class FileComparison:
    """
    Allows to compare or overwrite output files produced by tests.
    """

    @dataclass
    class LineDifference(Difference):
        """
        A difference between two corresponding lines in files.

        Attributes:
            line_number:    The line number where the difference occurs (starting at 1)
            expected_line:  The line that was expected
            actual_line:    The actual line
            file_name:      The names of the files that have been compared
        """
        line_number: int
        expected_line: str
        actual_line: str
        file_name: str

        def __str__(self) -> str:
            text = 'Found difference at line ' + str(self.line_number) + ' of file ' + self.file_name + '.\n\n'
            text += 'Expected:\n' + self.expected_line + '\n\n'
            text += 'Actual:\n' + self.actual_line
            return text

    ENV_OVERWRITE_OUTPUT_FILES = 'OVERWRITE_OUTPUT_FILES'

    @staticmethod
    def __should_overwrite_file() -> bool:
        value = environ.get(FileComparison.ENV_OVERWRITE_OUTPUT_FILES, 'false').strip().lower()

        if value == 'true':
            return True
        if value == 'false':
            return False
        raise ValueError('Value of environment variable "' + FileComparison.ENV_OVERWRITE_OUTPUT_FILES
                         + '" must be "true" or "false", but is "' + value + '"')

    @staticmethod
    def __replace_durations_with_placeholders(line: str) -> str:
        regex_duration = '(\\d+ (day(s)*|hour(s)*|minute(s)*|second(s)*|millisecond(s)*))'
        return regex.sub(regex_duration + '((, )' + regex_duration + ')*' + '(( and )' + regex_duration + ')?',
                         '<duration>', line)

    def __init__(self, lines: Iterable[str]):
        """
        :param lines: The lines in a file
        """
        self.lines = lines

    def compare_or_overwrite(self, another_file: str) -> Optional[Difference]:
        """
        Compares the file to another file or overwrites the latter with the former.

        :param another_file:    The path to a file to compare with or to be overwritten
        :return:                The first difference that has been found or None, if the files are the same
        """
        if self.__should_overwrite_file():
            makedirs(path.dirname(another_file), exist_ok=True)
            self._overwrite(self.lines, another_file)
            return None
        else:
            return self._compare(self.lines, another_file)

    def _compare(self, lines: Iterable[str], another_file: str) -> Optional[Difference]:
        """
        May be overridden by subclasses in order to compare to files.

        :param lines:           The lines of a file to be compared with another file
        :param another_file:    The path to another file to compare with
        :return:                The first difference that has been found or None, if the files are the same
        """
        with open(another_file, 'r', encoding=ENCODING_UTF8) as file:
            expected_lines = file.readlines()

            for i, actual_line in enumerate(lines):
                actual_line = self.__replace_durations_with_placeholders(actual_line.strip('\n'))
                expected_line = expected_lines[i].strip('\n')

                if actual_line != expected_line:
                    return FileComparison.LineDifference(line_number=i + 1,
                                                         actual_line=actual_line,
                                                         expected_line=expected_line,
                                                         file_name=path.basename(another_file))

    def _overwrite(self, lines: Iterable[str], another_file: str):
        """
        May be overridden by subclasses in order to overwrite a file with another one.

        :param lines:           The lines of a file
        :param another_file:    The path to another file that should be overwritten
        """
        with open(another_file, 'w+', encoding=ENCODING_UTF8) as file:
            for line in lines:
                line = self.__replace_durations_with_placeholders(line.strip('\n'))
                file.write(line + '\n')
