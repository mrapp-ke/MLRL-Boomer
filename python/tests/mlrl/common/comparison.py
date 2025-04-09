"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
import csv
import re as regex

from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import environ, makedirs, path
from typing import Any, Iterable, List, Optional, Set

from mlrl.testbed.io import CSV_DELIMITER, CSV_QUOTE_CHAR, ENCODING_UTF8

PLACEHOLDER_DURATION = '<duration>'


class Difference(ABC):
    """
    A difference between two files.
    """

    def __init__(self, file: str):
        """
        :param file: The path to the file that has been compared
        """
        self.file = file

    def __str__(self) -> str:
        return 'Found unexpected content in file "' + self.file + '"'


class FileComparison(ABC):
    """
    An abstract base class for all classes that allow to compare or overwrite output files produced by tests.
    """

    class LineDifference(Difference):
        """
        A difference between two corresponding lines in files.
        """

        def __init__(self, file: str, line_index: int, expected_line: str, actual_line: str):
            """
            :param file:            The path to the file that has been compared
            :param line_index:      The index of the line where the difference occurs
            :param expected_line:   The line that was expected
            :param actual_line:     The actual line
            """
            super().__init__(file)
            self.line_index = line_index
            self.expected_line = expected_line
            self.actual_line = actual_line

        def __str__(self) -> str:
            text = 'Line ' + str(self.line_index + 1) + ' is unexpected according to file "' + self.file + '".\n\n'
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
    def for_file(file: str) -> 'FileComparison':
        """
        Creates and returns a new object of type `FileComparison` for a specific file.

        :param file:    The path to the file
        :return:        The `FileComparison` that has been created
        """
        if file.endswith('.csv'):
            return CsvFileComparison(file)

        with open(file, mode='r', encoding=ENCODING_UTF8) as text_file:
            return TextFileComparison(text_file.readlines())

    def compare_or_overwrite(self, another_file: str) -> Optional[Difference]:
        """
        Compares the file to another file or overwrites the latter with the former.

        :param another_file:    The path to a file to compare with or to be overwritten
        :return:                The first difference that has been found or None, if the files are the same
        """
        if self.__should_overwrite_file():
            makedirs(path.dirname(another_file), exist_ok=True)
            self._overwrite(another_file)
            return None

        return self._compare(another_file)

    @abstractmethod
    def _compare(self, another_file: str) -> Optional[Difference]:
        """
        Must be implemented by subclasses in order to compare to files.

        :param another_file:    The path to another file to compare with
        :return:                The first difference that has been found or None, if the files are the same
        """

    @abstractmethod
    def _overwrite(self, another_file: str):
        """
        Must be implemented by subclasses in order to overwrite a file with another one.

        :param another_file: The path to another file that should be overwritten
        """


class TextFileComparison(FileComparison):
    """
    Allows to compare or overwrite text files produced by tests.
    """

    @staticmethod
    def __replace_durations_with_placeholders(line: str) -> str:
        regex_duration = '(\\d+ (day(s)*|hour(s)*|minute(s)*|second(s)*|millisecond(s)*))'
        return regex.sub(regex_duration + '((, )' + regex_duration + ')*' + '(( and )' + regex_duration + ')?',
                         PLACEHOLDER_DURATION, line)

    def __init__(self, lines: Iterable[str]):
        """
        :param lines: The lines in a file
        """
        self.lines = lines

    def _compare(self, another_file: str) -> Optional[Difference]:
        with open(another_file, 'r', encoding=ENCODING_UTF8) as file:
            expected_lines = file.readlines()

            for line_index, (actual_line, expected_line) in enumerate(zip(self.lines, expected_lines)):
                actual_line = self.__replace_durations_with_placeholders(actual_line.strip('\n'))
                expected_line = expected_line.strip('\n')

                if actual_line != expected_line:
                    return FileComparison.LineDifference(line_index=line_index,
                                                         actual_line=actual_line,
                                                         expected_line=expected_line,
                                                         file=another_file)

        return None

    def _overwrite(self, another_file: str):
        with open(another_file, 'w+', encoding=ENCODING_UTF8) as file:
            for line in self.lines:
                line = self.__replace_durations_with_placeholders(line.strip('\n'))
                file.write(line + '\n')


class CsvFileComparison(FileComparison):
    """
    Allows to compare or overwrite CSV files produced by tests.
    """

    class DimensionDifference(Difference):
        """
        A difference between two CSV files that have different numbers of rows or columns.
        """

        def __init__(self, file: str, num_expected_rows: int, num_expected_columns: int, num_actual_rows: int,
                     num_actual_columns: int):
            """
            :param file:                    The path to the file that has been compared
            :param num_expected_rows:       The expected number of rows
            :param num_expected_columns:    The expected number of columns
            :param num_actual_rows:         The actual number of rows
            :param num_actual_columns:      The actual number of columns
            """
            super().__init__(file)
            self.num_expected_rows = num_expected_rows
            self.num_expected_columns = num_expected_columns
            self.num_actual_rows = num_actual_rows
            self.num_actual_columns = num_actual_columns

        def __str__(self) -> str:
            return 'CSV file should have ' + str(self.num_expected_rows) + ' rows and ' + str(
                self.num_expected_columns) + ' columns according to file "' + self.file + '", but has ' + str(
                    self.num_actual_rows) + ' rows and ' + str(self.num_actual_columns) + ' columns'

    class CellDifferences(Difference):
        """
        A difference between two CSV files that contain different values for one or several cells.
        """

        @dataclass
        class CellDifference:
            """
            A difference between corresponding cells in two CSV files.

            Attributes:
                 row_index:         The index of the row, the cell belongs to
                 column_index:      The index of the column, the cell belongs to
                 expected_value:    The expected value in the cell
                 actual_value:      The actual value in the cell
                 header:            The header of the column, the cell belongs to
            """
            row_index: int
            column_index: int
            expected_value: Any
            actual_value: Any
            header: Any

            def __str__(self) -> str:
                return 'row ' + str(self.row_index
                                    + 1) + ', column ' + str(self.column_index + 1) + ' with header "' + str(
                                        self.header) + '": Value should be "' + str(
                                            self.expected_value) + '", but is "' + str(self.actual_value) + '"'

        def __init__(self, file: str, different_cells: List[CellDifference]):
            """
            :param file:            The path to the file that has been compared
            :param different_cells: A list that contains all cells with unexpected values
            """
            super().__init__(file)
            self.different_cells = different_cells

        def __str__(self) -> str:
            different_cells = self.different_cells
            text = 'Found ' + str(len(different_cells)) + ' unexpected ' + (
                'value' if len(different_cells) == 1 else 'values') + ' according to file "' + self.file + '":\n\n'

            for cell in different_cells:
                text += str(cell) + '\n'

            return text

    def __init__(self, file: str):
        """
        :param file: The path to a file
        """
        self.file = file

    def __get_duration_column_indices(self, headers: List[Any]) -> Set[int]:
        return {column_index for column_index, header in enumerate(headers) if 'time' in header.lower().split()}

    def _compare(self, another_file: str) -> Optional[Difference]:
        with open(self.file, mode='r', encoding=ENCODING_UTF8) as actual_file:
            actual_csv_file = csv.reader(actual_file, delimiter=CSV_DELIMITER, quotechar=CSV_QUOTE_CHAR)
            num_actual_rows = sum(1 for _ in actual_csv_file)
            actual_file.seek(0)
            num_actual_columns = len(next(actual_csv_file))

            with open(another_file, mode='r', encoding=ENCODING_UTF8) as expected_file:
                expected_csv_file = csv.reader(expected_file, delimiter=CSV_DELIMITER, quotechar=CSV_QUOTE_CHAR)
                num_expected_rows = sum(1 for _ in expected_csv_file)
                expected_file.seek(0)
                headers = next(expected_csv_file)
                num_expected_columns = len(headers)

                if num_actual_rows != num_expected_rows or num_actual_columns != num_expected_columns:
                    return CsvFileComparison.DimensionDifference(file=another_file,
                                                                 num_expected_rows=num_expected_rows,
                                                                 num_expected_columns=num_expected_columns,
                                                                 num_actual_rows=num_actual_rows,
                                                                 num_actual_columns=num_actual_columns)

                actual_file.seek(0)
                expected_file.seek(0)
                duration_column_indices = self.__get_duration_column_indices(headers)
                different_cells = []

                for row_index, (actual_row, expected_row) in enumerate(zip(actual_csv_file, expected_csv_file)):
                    for column_index, (actual_value, expected_value) in enumerate(zip(actual_row, expected_row)):
                        if row_index > 0 and column_index in duration_column_indices:
                            actual_value = PLACEHOLDER_DURATION

                        if actual_value != expected_value:
                            different_cells.append(
                                CsvFileComparison.CellDifferences.CellDifference(row_index=row_index,
                                                                                 column_index=column_index,
                                                                                 expected_value=expected_value,
                                                                                 actual_value=actual_value,
                                                                                 header=headers[column_index]))

                if different_cells:
                    return CsvFileComparison.CellDifferences(file=another_file, different_cells=different_cells)

        return None

    def _overwrite(self, another_file: str):
        with open(self.file, 'r', encoding=ENCODING_UTF8) as input_file:
            input_csv_file = csv.reader(input_file, delimiter=CSV_DELIMITER, quotechar=CSV_QUOTE_CHAR)
            headers = next(input_csv_file)
            duration_column_indices = self.__get_duration_column_indices(headers)

            with open(another_file, 'w+', encoding=ENCODING_UTF8) as output_file:
                output_csv_file = csv.writer(output_file,
                                             delimiter=CSV_DELIMITER,
                                             quotechar=CSV_QUOTE_CHAR,
                                             quoting=csv.QUOTE_MINIMAL,
                                             lineterminator='\n')
                output_csv_file.writerow(headers)

                for row in input_csv_file:
                    for column_index in duration_column_indices:
                        row[column_index] = PLACEHOLDER_DURATION

                    output_csv_file.writerow(row)
