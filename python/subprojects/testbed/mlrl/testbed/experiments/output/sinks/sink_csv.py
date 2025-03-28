"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing output data to CSV files.
"""
from mlrl.common.config.options import Options

from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks.sink import FileSink
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.io import SUFFIX_CSV, open_writable_file
from mlrl.testbed.util.io_csv import CsvWriter


class CsvFileSink(FileSink):
    """
    Allows to write output data to a CSV file.
    """

    def __init__(self, directory: str, options: Options = Options()):
        """
        :param directory:   The path to the directory of the file
        :param options:     Options to be taken into account
        """
        super().__init__(directory=directory, suffix=SUFFIX_CSV)
        self.options = options

    def _write_to_file(self, file_path: str, state: ExperimentState, output_data: OutputData, **kwargs):
        tabular_data = output_data.to_table(self.options, **kwargs)

        if tabular_data:
            prediction_result = state.prediction_result
            incremental_prediction = prediction_result and not prediction_result.prediction_scope.is_global

            if incremental_prediction:
                for row in tabular_data:
                    row['Model size'] = prediction_result.prediction_scope.model_size

            if tabular_data:
                header = sorted(tabular_data[0].keys())

                with open_writable_file(file_path, append=incremental_prediction) as csv_file:
                    csv_writer = CsvWriter(csv_file, header)

                    for row in tabular_data:
                        csv_writer.writerow(row)
