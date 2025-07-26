"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing output data to CSV files.
"""
import csv

from pathlib import Path
from typing import override

from mlrl.testbed.experiments.output.data import OutputValue
from mlrl.testbed.experiments.output.sinks.sink import TabularFileSink
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.experiments.table import Table
from mlrl.testbed.util.io import open_writable_file

from mlrl.util.options import Options


class CsvFileSink(TabularFileSink):
    """
    Allows to write tabular output data to a CSV file.
    """

    SUFFIX_CSV = 'csv'

    DELIMITER = ','

    QUOTE_CHAR = '"'

    def __init__(self, directory: Path, options: Options = Options(), create_directory: bool = False):
        """
        :param directory:           The path to the directory of the file
        :param options:             Options to be taken into account
        :param create_directory:    True, if the given directory should be created, if it does not exist, False
                                    otherwise
        """
        super().__init__(directory=directory,
                         suffix=self.SUFFIX_CSV,
                         options=options,
                         create_directory=create_directory)

    @override
    def _write_table_to_file(self, file_path: Path, state: ExperimentState, table: Table, **_):
        table = table.to_column_wise_table()
        prediction_result = state.prediction_result
        incremental_prediction = False

        if prediction_result:
            incremental_prediction = not prediction_result.prediction_scope.is_global

            if incremental_prediction:
                model_size = prediction_result.prediction_scope.model_size
                table.add_column(*[model_size for _ in range(table.num_rows)],
                                 header=OutputValue('model_size', 'Model size'))

        table.sort_by_headers()

        with open_writable_file(file_path, append=incremental_prediction) as csv_file:
            csv_writer = csv.writer(csv_file,
                                    delimiter=self.DELIMITER,
                                    quotechar=self.QUOTE_CHAR,
                                    quoting=csv.QUOTE_MINIMAL)

            if csv_file.tell() == 0:
                header_row = table.header_row

                if header_row:
                    csv_writer.writerow(header_row)

            for row in table.rows:
                csv_writer.writerow(row)
