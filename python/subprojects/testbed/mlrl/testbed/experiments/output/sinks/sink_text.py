"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing output data to text files.
"""
from mlrl.testbed.experiments.output.sinks.sink import FileSink
from mlrl.testbed.experiments.state import ExperimentState
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
                     formatter_options: ExperimentState.FormatterOptions = ExperimentState.FormatterOptions()):
            super().__init__(directory, file_name, 'txt', formatter_options)

    def _write_to_file(self, file_path: str, _: ExperimentState, output_data, **kwargs):
        text = output_data.to_text(self.options, **kwargs)

        if text:
            with open_writable_file(file_path) as text_file:
                text_file.write(text)
