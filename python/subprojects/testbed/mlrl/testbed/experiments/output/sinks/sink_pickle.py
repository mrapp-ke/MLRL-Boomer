"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing output data to files by using Python's pickle mechanism.
"""
import pickle

from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks.sink import FileSink
from mlrl.testbed.experiments.state import ExperimentState

from mlrl.util.options import Options


class PickleFileSink(FileSink):
    """
    Allows to write output data to a file by using Python's pickle mechanism.
    """

    SUFFIX_PICKLE = 'pickle'

    def __init__(self, directory: str, options: Options = Options(), create_directory: bool = False):
        """
        :param directory:           The path to the directory of the file
        :param options:             Options to be taken into account
        :param create_directory:    True, if the given directory should be created, if it does not exist, False
                                    otherwise
        """
        super().__init__(directory=directory,
                         suffix=self.SUFFIX_PICKLE,
                         options=options,
                         create_directory=create_directory)

    # pylint: disable=unused-argument
    def _write_to_file(self, file_path: str, state: ExperimentState, output_data: OutputData, **_):
        output_object = output_data.to_object()

        if output_object:
            with open(file_path, mode='wb') as pickle_file:
                pickle.dump(output_object, pickle_file, pickle.HIGHEST_PROTOCOL)
