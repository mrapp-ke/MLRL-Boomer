"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for loading and printing parameter settings that are used by a learner. The parameter settings can be
written to one or several outputs, e.g., to the console or to a file. They can also be loaded from CSV files.
"""
import logging as log

from abc import ABC, abstractmethod
from os import path
from typing import Any, Dict, Optional

from mlrl.common.config.options import Options

from mlrl.testbed.experiments.output.converters import TableConverter, TextConverter
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink as BaseCsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink as BaseLogSink
from mlrl.testbed.experiments.output.writer import OutputWriter
from mlrl.testbed.experiments.state import ExperimentState, PredictionResult
from mlrl.testbed.fold import Fold
from mlrl.testbed.format import format_table
from mlrl.testbed.util.io import SUFFIX_CSV, get_file_name_per_fold, open_readable_file
from mlrl.testbed.util.io_csv import CsvReader


class ParameterLoader(ABC):
    """
    An abstract base class for all classes that load parameters from an input.
    """

    @abstractmethod
    def load_parameters(self, fold: Fold) -> Dict:
        """
        Loads a parameter setting from the input.

        :param fold:    The fold of the available data, the parameter setting corresponds to
        :return:        A dictionary that stores the parameters
        """


class CsvParameterLoader(ParameterLoader):
    """
    Loads parameter settings from CSV files.
    """

    def __init__(self, input_dir: str):
        """
        :param input_dir: The path to the directory, the CSV files should be read from
        """
        self.input_dir = input_dir

    def load_parameters(self, fold: Fold) -> Dict:
        file_name = get_file_name_per_fold('parameters', SUFFIX_CSV, fold.index)
        file_path = path.join(self.input_dir, file_name)
        log.debug('Loading parameters from file \"%s\"...', file_path)

        try:
            with open_readable_file(file_path) as csv_file:
                csv_reader = CsvReader(csv_file)
                log.info('Successfully loaded parameters from file \"%s\"', file_path)
                return dict(next(csv_reader))
        except IOError:
            log.error('Failed to load parameters from file \"%s\"', file_path)
            return {}


class ParameterWriter(OutputWriter):
    """
    Allows to write parameter settings to one or several sinks.
    """

    class Parameters(TextConverter, TableConverter):
        """
        Stores the parameter settings of a learner.
        """

        def __init__(self, parameters: Dict[str, Any]):
            """
            :param parameters: A dictionary that stores the parameters
            """
            self.parameters = parameters

        # pylint: disable=unused-argument
        def to_text(self, options: Options, **_) -> Optional[str]:
            """
            See :func:`mlrl.testbed.experiments.output.converters.TextConverter.to_text`
            """
            parameters = self.parameters
            rows = []

            for key in sorted(parameters):
                value = parameters[key]

                if value is not None:
                    rows.append([str(key), str(value)])

            return format_table(rows)

        # pylint: disable=unused-argument
        def to_table(self, options: Options, **_) -> Optional[TableConverter.Table]:
            """
            See :func:`mlrl.testbed.experiments.output.converters.TableConverter.to_table`
            """
            parameters = self.parameters
            columns = {}

            for key, value in parameters.items():
                if value is not None:
                    columns[key] = value

            return [columns]

    class LogSink(BaseLogSink):
        """
        Allows to write parameter settings to the console.
        """

        def __init__(self):
            super().__init__(BaseLogSink.TitleFormatter('Custom parameters', include_dataset_type=False))

    class CsvFileSink(BaseCsvFileSink):
        """
        Allows to write parameter settings to CSV files.
        """

        def __init__(self, directory: str):
            """
            :param directory: The path to the directory, where the CSV file should be located
            """
            super().__init__(BaseCsvFileSink.PathFormatter(directory, 'parameters', include_dataset_type=False))

    def _generate_output_data(self, state: ExperimentState, _: Optional[PredictionResult]) -> Optional[Any]:
        return ParameterWriter.Parameters(state.parameters)
