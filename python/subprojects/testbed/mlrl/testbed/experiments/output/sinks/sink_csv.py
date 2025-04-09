"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing output data to CSV files.
"""
from csv import QUOTE_MINIMAL, DictWriter

from mlrl.common.config.options import Options

from mlrl.testbed.experiments.output.data import TabularOutputData
from mlrl.testbed.experiments.output.sinks.sink import TabularFileSink
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.io import open_writable_file


class CsvFileSink(TabularFileSink):
    """
    Allows to write tabular output data to a CSV file.
    """

    SUFFIX_CSV = 'csv'

    DELIMITER = ','

    QUOTE_CHAR = '"'

    def __init__(self, directory: str, options: Options = Options()):
        """
        :param directory:   The path to the directory of the file
        :param options:     Options to be taken into account
        """
        super().__init__(directory=directory, suffix=self.SUFFIX_CSV, options=options)

    def _write_table_to_file(self, file_path: str, state: ExperimentState, table: TabularOutputData.Table, **_):
        prediction_result = state.prediction_result
        incremental_prediction = prediction_result and not prediction_result.prediction_scope.is_global

        if incremental_prediction:
            for row in table:
                row['Model size'] = prediction_result.prediction_scope.model_size

        header = sorted(table[0].keys())

        with open_writable_file(file_path, append=incremental_prediction) as csv_file:
            dict_writer = DictWriter(csv_file,
                                     delimiter=self.DELIMITER,
                                     quotechar=self.QUOTE_CHAR,
                                     quoting=QUOTE_MINIMAL,
                                     fieldnames=header)

            for row in table:
                dict_writer.writerow(row)
