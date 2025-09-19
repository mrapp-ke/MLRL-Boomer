"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow reading input data from CSV files.
"""
import csv

from pathlib import Path
from typing import Optional, override

from mlrl.testbed.experiments.input.data import TabularInputData
from mlrl.testbed.experiments.input.sources.source import TabularFileSource
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.table import RowWiseTable, Table
from mlrl.testbed.util.io import open_readable_file


class CsvFileSource(TabularFileSource):
    """
    Allows to read tabular input data from a CSV file.
    """

    def __init__(self, directory: Path):
        """
        :param directory: The path to the directory of the file
        """
        super().__init__(directory=directory, suffix=CsvFileSink.SUFFIX_CSV)

    @override
    def _read_table_from_file(self, file_path: Path, input_data: TabularInputData) -> Optional[Table]:
        with open_readable_file(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=CsvFileSink.DELIMITER, quotechar=CsvFileSink.QUOTE_CHAR)
            properties = input_data.properties
            has_header = isinstance(properties, TabularInputData.Properties) and properties.has_header

            try:
                header_row = next(csv_reader) if has_header else []
                table = RowWiseTable(*header_row)
            except StopIteration:
                table = RowWiseTable()

            for row in csv_reader:
                table.add_row(*row)

            return table
