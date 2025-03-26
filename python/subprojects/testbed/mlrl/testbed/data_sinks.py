"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement sinks, output data can be written to.
"""
import logging as log

from typing import Optional

from mlrl.common.config.options import Options

from mlrl.testbed.io import SUFFIX_CSV, SUFFIX_TEXT, create_csv_dict_writer, open_writable_csv_file, \
    open_writable_text_file
from mlrl.testbed.output.sinks.sink import FileSink, Sink
from mlrl.testbed.output_scope import OutputScope
from mlrl.testbed.prediction_result import PredictionResult
from mlrl.testbed.training_result import TrainingResult


class LogSink(Sink):
    """
    Allows to write output data to the console.
    """

    class TitleFormatter:
        """
        Allows to format the title that is printed before the output data.
        """

        def __init__(self,
                     title: str,
                     include_dataset_type: bool = True,
                     include_prediction_scope: bool = True,
                     include_fold: bool = True):
            """
            :param title:                       A title
            :param include_dataset_type:        True, if the type of the dataset should be included in the title, False
                                                otherwise
            :param include_prediction_scope:    True, if the scope of the predictions should be included in the title,
                                                False otherwise
            :param include_fold:                True, if the cross validation fold should be included in the title,
                                                False otherwise
            """
            self.title = title
            self.include_dataset_type = include_dataset_type
            self.include_prediction_scope = include_prediction_scope
            self.include_fold = include_fold

        def __format_dataset_type(self, scope: OutputScope) -> str:
            if self.include_dataset_type:
                return ' for ' + scope.dataset.type.value + ' data'
            return ''

        def __format_fold(self, scope: OutputScope) -> str:
            if self.include_fold:
                fold = scope.fold

                if fold.is_cross_validation_used:
                    if fold.index is None:
                        formatted_fold = 'Average across ' + str(fold.num_folds) + ' folds'
                    else:
                        formatted_fold = 'Fold ' + str(fold.index + 1)

                    return ' (' + formatted_fold + ')'

            return ''

        def __format_prediction_scope(self, prediction_result: Optional[PredictionResult]) -> str:
            if self.include_prediction_scope and prediction_result:
                prediction_scope = prediction_result.prediction_scope

                if not prediction_scope.is_global():
                    return ' using a model of size ' + str(prediction_scope.get_model_size())

            return ''

        def format(self, scope: OutputScope, prediction_result: Optional[PredictionResult]) -> str:
            """
            Formats and returns the title that is printed before the output data.

            :param scope:               The scope of the output data
            :param prediction_result:   A `PredictionResult` that stores the result of a prediction process or None, if
                                        no predictions have been obtained
            """
            return self.title + self.__format_dataset_type(scope) + self.__format_prediction_scope(
                prediction_result) + self.__format_fold(scope)

    def __init__(self, title_formatter: TitleFormatter, options: Options = Options()):
        """
        :param title_formatter: A `TitleFormatter` to be used for formatting the title that is printed before the
                                output data
        """
        super().__init__(options=options)
        self.title_formatter = title_formatter

    def write_to_sink(self, scope: OutputScope, _: Optional[TrainingResult],
                      prediction_result: Optional[PredictionResult], output_data, **kwargs):
        """
        See :func:`mlrl.testbed.output.sinks.sink.Sink.write_to_sink`
        """
        log.info('%s:\n\n%s\n', self.title_formatter.format(scope, prediction_result),
                 output_data.to_text(self.options, **kwargs))


class TextFileSink(FileSink):
    """
    Allows to write output data to a text file.
    """

    class PathFormatter(FileSink.PathFormatter):
        """
        Allows to determine the paths to text files to which output data is written.
        """

        def __init__(self,
                     directory: str,
                     file_name: str,
                     include_dataset_type: bool = True,
                     include_prediction_scope: bool = True,
                     include_fold: bool = True):
            super().__init__(directory, file_name, SUFFIX_TEXT, include_dataset_type, include_prediction_scope,
                             include_fold)

    # pylint: disable=unused-argument
    def _write_output(self, file_path: str, scope: OutputScope, training_result: Optional[TrainingResult],
                      prediction_result: Optional[PredictionResult], output_data, **kwargs):
        with open_writable_text_file(file_path) as text_file:
            text_file.write(output_data.to_text(self.options, **kwargs))


class CsvFileSink(FileSink):
    """
    Allows to write output data to a CSV file.
    """

    class PathFormatter(FileSink.PathFormatter):
        """
        Allows to determine the paths to text files to which output data is written.
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
    def _write_output(self, file_path: str, scope: OutputScope, training_result: Optional[TrainingResult],
                      prediction_result: Optional[PredictionResult], output_data, **kwargs):
        tabular_data = output_data.to_table(self.options, **kwargs)

        if tabular_data:
            incremental_prediction = prediction_result and not prediction_result.prediction_scope.is_global()

            if incremental_prediction:
                for row in tabular_data:
                    row['Model size'] = prediction_result.prediction_scope.get_model_size()

            if tabular_data:
                header = sorted(tabular_data[0].keys())

                with open_writable_csv_file(file_path, append=incremental_prediction) as csv_file:
                    csv_writer = create_csv_dict_writer(csv_file, header)

                    for row in tabular_data:
                        csv_writer.writerow(row)
