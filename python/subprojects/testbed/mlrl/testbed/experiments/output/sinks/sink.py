"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing sinks, output data may be written to.
"""
from abc import ABC, abstractmethod
from os import path
from typing import Optional

from mlrl.common.config.options import Options

from mlrl.testbed.output_scope import OutputScope
from mlrl.testbed.prediction_result import PredictionResult
from mlrl.testbed.training_result import TrainingResult
from mlrl.testbed.util.io import get_file_name_per_fold


class Sink(ABC):
    """
    An abstract base class for all sinks, output data may be written to.
    """

    def __init__(self, options: Options = Options()):
        """
        :param options: Options to be taken into account
        """
        self.options = options

    @abstractmethod
    def write_to_sink(self, scope: OutputScope, training_result: Optional[TrainingResult],
                      prediction_result: Optional[PredictionResult], output_data, **kwargs):
        """
        Must be implemented by subclasses in order to write output data to the sink.

        :param scope:               The scope of the output data
        :param training_result:     A `TrainingResult` that stores the result of a training process or None, if no
                                    model has been trained
        :param prediction_result:   A `PredictionResult` that stores the result of a prediction process or None, if
                                    no predictions have been obtained
        :param output_data:         The output data that should be written to the sink
        """


class FileSink(Sink, ABC):
    """
    An abstract base class for all sinks that write output data to a file.
    """

    class PathFormatter:
        """
        Allows to determine the path to the file to which output data is written.
        """

        def __init__(self,
                     directory: str,
                     file_name: str,
                     suffix: str,
                     include_dataset_type: bool = True,
                     include_prediction_scope: bool = True,
                     include_fold: bool = True):
            """
            :param directory:                   The path to the directory of the file
            :param file_name:                   The name of the file
            :param suffix:                      The suffix of the file
            :param include_dataset_type:        True, if the type of the dataset should be included in the file name,
                                                False otherwise
            :param include_prediction_scope:    True, if the scope of the predictions should be included in the file
                                                name, False otherwise
            :param include_fold:                True, if the cross validation fold should be included in the file name,
                                                False otherwise
            """
            self.directory = directory
            self.file_name = file_name
            self.suffix = suffix
            self.include_dataset_type = include_dataset_type
            self.include_prediction_scope = include_prediction_scope
            self.include_fold = include_fold

        def format(self, scope: OutputScope, prediction_result: Optional[PredictionResult]) -> str:
            """
            Determines and returns the path to the file to which the output data should be written.

            :param scope:               The scope of the output data
            :param prediction_result:   A `PredictionResult` that stores the result of a prediction process or None, if
                                        no predictions have been obtained
            """
            file_name = self.file_name

            if self.include_dataset_type:
                file_name = scope.dataset.type.get_file_name(file_name)

            if self.include_prediction_scope and prediction_result:
                file_name = prediction_result.prediction_scope.get_file_name(file_name)

            if self.include_fold:
                file_name = get_file_name_per_fold(file_name, self.suffix, scope.fold.index)

            return path.join(self.directory, file_name)

    def __init__(self, path_formatter: PathFormatter, options: Options = Options()):
        """
        :param: path_formatter: A `PathFormatter` to be used for determining the path to the file to which output data
                                should be written
        """
        super().__init__(options)
        self.path_formatter = path_formatter

    def write_to_sink(self, scope: OutputScope, training_result: Optional[TrainingResult],
                      prediction_result: Optional[PredictionResult], output_data, **kwargs):
        """
        See :func:`mlrl.testbed.experiments.output.sinks.sink.Sink.write_to_sink`
        """
        file_path = self.path_formatter.format(scope, prediction_result)
        self._write_to_file(file_path, scope, training_result, prediction_result, output_data, **kwargs)

    @abstractmethod
    def _write_to_file(self, file_path: str, scope: OutputScope, training_result: Optional[TrainingResult],
                       prediction_result: Optional[PredictionResult], output_data, **kwargs):
        """
        Must be implemented by subclasses in order to write output data to a specific file.

        :param file_path:           The path to the file to which the output data should be written
        :param scope:               The scope of the output data
        :param training_result:     A `TrainingResult` that stores the result of a training process or None, if no
                                    model has been trained
        :param prediction_result:   A `PredictionResult` that stores the result of a prediction process or None, if
                                    no predictions have been obtained
        :param output_data:         The output data that should be written to the file
        """
