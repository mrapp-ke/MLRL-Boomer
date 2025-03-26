"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for reading and writing CSV files.
"""
from csv import QUOTE_MINIMAL, DictReader, DictWriter
from typing import Collection

DELIMITER = ','

QUOTE_CHAR = '"'


class CsvReader(DictReader):
    """
    Allows reading from a CSV file.
    """

    def __init__(self, csv_file):
        """
        :param csv_file: The CSV file
        """
        super().__init__(csv_file, delimiter=DELIMITER, quotechar=QUOTE_CHAR)


class CsvWriter(DictWriter):
    """
    Allows writing to a CSV file.
    """

    def __init__(self, csv_file, header: Collection[str]):
        """
        :param csv_file:    The CSV file
        :param header:      The header of the CSV file
        """
        super().__init__(csv_file, delimiter=DELIMITER, quotechar=QUOTE_CHAR, quoting=QUOTE_MINIMAL, fieldnames=header)
        if csv_file.mode == 'w':
            self.writeheader()
