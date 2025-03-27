"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing output data to text files.
"""
from typing import Optional

from mlrl.testbed.experiments.output.sinks.sink import FileSink
from mlrl.testbed.experiments.state import ExperimentState, PredictionResult
from mlrl.testbed.util.io import open_writable_file


class TextFileSink(FileSink):
    """
    Allows to write output data to a text file.
    """

    class PathFormatter(FileSink.PathFormatter):
        """
        Allows to determine the path to the text file to which output data is written.
        """

        def __init__(self,
                     directory: str,
                     file_name: str,
                     include_dataset_type: bool = True,
                     include_prediction_scope: bool = True,
                     include_fold: bool = True):
            super().__init__(directory, file_name, 'txt', include_dataset_type, include_prediction_scope, include_fold)

    # pylint: disable=unused-argument
    def _write_to_file(self, file_path: str, state: ExperimentState, prediction_result: Optional[PredictionResult],
                       output_data, **kwargs):
        text = output_data.to_text(self.options, **kwargs)

        if text:
            with open_writable_file(file_path) as text_file:
                text_file.write(text)
