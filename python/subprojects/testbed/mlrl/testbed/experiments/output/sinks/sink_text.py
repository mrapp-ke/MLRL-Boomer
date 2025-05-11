"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing output data to text files.
"""
from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks.sink import FileSink
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.util.io import open_writable_file

from mlrl.util.options import Options


class TextFileSink(FileSink):
    """
    Allows to write textual output data to a text file.
    """

    def __init__(self, directory: str, options: Options = Options(), create_directory: bool = False):
        """
        :param directory:           The path to the directory of the file
        :param options:             Options to be taken into account
        :param create_directory:    True, if the given directory should be created, if it does not exist, False
                                    otherwise
        """
        super().__init__(directory=directory, suffix='txt', options=options, create_directory=create_directory)

    def _write_to_file(self, file_path: str, _: ExperimentState, output_data: OutputData, **kwargs):
        text = output_data.to_text(self.options, **kwargs)

        if text:
            with open_writable_file(file_path) as text_file:
                text_file.write(text)
