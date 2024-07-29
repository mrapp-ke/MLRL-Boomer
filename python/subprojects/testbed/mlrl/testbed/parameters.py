"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for loading and printing parameter settings that are used by a learner. The parameter settings can be
written to one or several outputs, e.g., to the console or to a file. They can also be loaded from CSV files.
"""
import logging as log

from abc import ABC, abstractmethod
from os import path
from typing import Any, Dict, List, Optional

from mlrl.common.options import Options

from mlrl.testbed.data import MetaData
from mlrl.testbed.data_splitting import DataSplit, DataType
from mlrl.testbed.format import format_table
from mlrl.testbed.io import SUFFIX_CSV, create_csv_dict_reader, get_file_name_per_fold, open_readable_csv_file
from mlrl.testbed.output_writer import Formattable, OutputWriter, Tabularizable
from mlrl.testbed.prediction_scope import PredictionScope, PredictionType
from mlrl.testbed.problem_type import ProblemType


class ParameterInput(ABC):
    """
    An abstract base class for all classes that read parameters from an input.
    """

    @abstractmethod
    def read_parameters(self, data_split: DataSplit) -> Dict:
        """
        Reads a parameter setting from the input.

        :param data_split:  Information about the split of the available data, the parameter setting corresponds to
        :return:            A dictionary that stores the parameters
        """


class ParameterCsvInput(ParameterInput):
    """
    Reads parameter settings from CSV files.
    """

    def __init__(self, input_dir: str):
        """
        :param input_dir: The path of the directory, the CSV files should be read from
        """
        self.input_dir = input_dir

    def read_parameters(self, data_split: DataSplit) -> Dict:
        file_name = get_file_name_per_fold('parameters', SUFFIX_CSV, data_split.get_fold())
        file_path = path.join(self.input_dir, file_name)
        log.debug('Loading parameters from file \"%s\"...', file_path)

        try:
            with open_readable_csv_file(file_path) as csv_file:
                csv_reader = create_csv_dict_reader(csv_file)
                log.info('Successfully loaded parameters from file \"%s\"', file_path)
                return dict(next(csv_reader))
        except IOError:
            log.error('Failed to load parameters from file \"%s\"', file_path)
            return {}


class ParameterWriter(OutputWriter):
    """
    Allows to write parameter settings to one or several sinks.
    """

    class Parameters(Formattable, Tabularizable):
        """
        Stores the parameter settings of a learner.
        """

        def __init__(self, learner):
            """
            :param learner: A learner
            """
            self.params = learner.get_params()

        # pylint: disable=unused-argument
        def format(self, options: Options, **_):
            """
            See :func:`mlrl.testbed.output_writer.Formattable.format`
            """
            params = self.params
            rows = []

            for key in sorted(params):
                value = params[key]

                if value is not None:
                    rows.append([str(key), str(value)])

            return format_table(rows)

        # pylint: disable=unused-argument
        def tabularize(self, options: Options, **_) -> Optional[List[Dict[str, str]]]:
            """
            See :func:`mlrl.testbed.output_writer.Tabularizable.tabularize`
            """
            params = self.params
            columns = {}

            for key, value in params.items():
                if value is not None:
                    columns[key] = value

            return [columns]

    class LogSink(OutputWriter.LogSink):
        """
        Allows to write parameter settings to the console.
        """

        def __init__(self):
            super().__init__(title='Custom parameters')

    class CsvSink(OutputWriter.CsvSink):
        """
        Allows to write parameter settings to CSV files.
        """

        def __init__(self, output_dir: str):
            super().__init__(output_dir=output_dir, file_name='parameters')

    # pylint: disable=unused-argument
    def _generate_output_data(self, problem_type: ProblemType, meta_data: MetaData, x, y, data_split: DataSplit,
                              learner, data_type: Optional[DataType], prediction_type: Optional[PredictionType],
                              prediction_scope: Optional[PredictionScope], predictions: Optional[Any],
                              train_time: float, predict_time: float) -> Optional[Any]:
        return ParameterWriter.Parameters(learner)
