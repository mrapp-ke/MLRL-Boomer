"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing output data to CSV files.
"""
from typing import Optional

from mlrl.testbed.output.sinks.sink import FileSink
from mlrl.testbed.output_scope import OutputScope
from mlrl.testbed.prediction_result import PredictionResult
from mlrl.testbed.training_result import TrainingResult
from mlrl.testbed.util.io import SUFFIX_CSV, open_writable_file
from mlrl.testbed.util.io_csv import CsvWriter


class CsvFileSink(FileSink):
    """
    Allows to write output data to a CSV file.
    """

    class PathFormatter(FileSink.PathFormatter):
        """
        Allows to determine the path to the CSV file to which output data is written.
        """

        def __init__(self,
                     directory: str,
                     file_name: str,
                     include_dataset_type: bool = True,
                     include_prediction_scope: bool = True,
                     include_fold: bool = True):
            super().__init__(directory, file_name, SUFFIX_CSV, include_dataset_type, include_prediction_scope,
                             include_fold)

    # pylint: disable=unused-argument
    def _write_to_file(self, file_path: str, scope: OutputScope, training_result: Optional[TrainingResult],
                       prediction_result: Optional[PredictionResult], output_data, **kwargs):
        tabular_data = output_data.to_table(self.options, **kwargs)

        if tabular_data:
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
