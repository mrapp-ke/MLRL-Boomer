"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow writing output data to files by using Python's pickle mechanism.
"""
import pickle

from mlrl.common.config.options import Options

from mlrl.testbed.experiments.output.data import OutputData
from mlrl.testbed.experiments.output.sinks.sink import FileSink
from mlrl.testbed.experiments.state import ExperimentState


class PickleFileSink(FileSink):
    """
    Allows to write output data to a file by using Python's pickle mechanism.
    """

    SUFFIX_PICKLE = 'pickle'

    def __init__(self, directory: str, options: Options = Options()):
        """
        :param directory:   The path to the directory of the file
        :param options:     Options to be taken into account
        """
        super().__init__(directory=directory, suffix=self.SUFFIX_PICKLE, options=options)

    # pylint: disable=unused-argument
    def _write_to_file(self, file_path: str, state: ExperimentState, output_data: OutputData, **_):
        output_object = output_data.to_object(self.options)

        if output_object:
            with open(file_path, mode='wb') as pickle_file:
                pickle.dump(output_object, pickle_file, pickle.HIGHEST_PROTOCOL)
