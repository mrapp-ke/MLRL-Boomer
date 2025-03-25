"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for writing output data to sinks like the console or output files.
"""
import logging as log

from abc import ABC, abstractmethod
from os import path
from typing import Any, Dict, List, Optional

from mlrl.common.config.options import Options

from mlrl.testbed.io import SUFFIX_CSV, create_csv_dict_writer, get_file_name_per_fold, open_writable_csv_file, \
    open_writable_text_file
from mlrl.testbed.output_scope import OutputScope
from mlrl.testbed.prediction_result import PredictionResult
from mlrl.testbed.training_result import TrainingResult


class Formattable(ABC):
    """
    An abstract base class for all classes from which a textual representation can be created.
    """

    @abstractmethod
    def format(self, options: Options, **kwargs) -> str:
        """
        Creates and returns a textual representation of the object.

        :param options: Options to be taken into account
        :return:        The textual representation that has been created
        """


class Tabularizable(ABC):
    """
    An abstract base class for all classes from which a tabular representation can be created.
    """

    @abstractmethod
    def tabularize(self, options: Options, **kwargs) -> Optional[List[Dict[str, str]]]:
        """
        Creates and returns a tabular representation of the object.

        :param options: Options to be taken into account
        :return:        The tabular representation that has been created
        """


class OutputWriter(ABC):
    """
    An abstract base class for all classes that allow to write output data to one or several sinks, e.g., the console or
    output files.
    """

    KWARG_DATA_SPLIT = 'data_split'

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
        def write_output(self, scope: OutputScope, prediction_result: Optional[PredictionResult], output_data,
                         **kwargs):
            """
            Must be implemented by subclasses in order to write output data to the sink.

            :param scope:               The scope of the output data
            :param prediction_result:   A `PredictionResult` that provides access to predictions or None, if no
                                        predictions have been obtained
                                        incrementally or None, if no predictions have been obtained
            :param output_data:         The output data that should be written to the sink
            """

    class LogSink(Sink):
        """
        Allows to write output data to the console.
        """

        def __init__(self, title: str, options: Options = Options()):
            """
            :param title:   A title that is printed before the actual output data
            """
            super().__init__(options=options)
            self.title = title

        def write_output(self, scope: OutputScope, prediction_result: Optional[PredictionResult], output_data,
                         **kwargs):
            message = self.title
            dataset_type = scope.dataset.type

            if dataset_type:
                message += ' for ' + dataset_type.value + ' data'

            prediction_scope = prediction_result.prediction_scope if prediction_result else None

            if prediction_scope and not prediction_scope.is_global():
                message += ' using a model of size ' + str(prediction_scope.get_model_size())

            fold = scope.fold

            if fold.is_cross_validation_used:
                message += ' ('
                message += 'Average across ' + str(
                    fold.num_folds) + ' folds' if fold.index is None else 'Fold ' + str(fold.index + 1)
                message += ')'

            message += ':\n\n' + output_data.format(self.options, **kwargs) + '\n'
            log.info(message)

    class TextFileSink(Sink):
        """
        Allows to write output data to a text file.
        """

        def __init__(self, output_dir: str, file_name: str, options: Options = Options()):
            """
            :param output_dir:  The path to the directory, where the text file should be located
            :param file_name:   The name of the text file (without suffix)
            """
            super().__init__(options=options)
            self.output_dir = output_dir
            self.file_name = file_name

        def write_output(self, scope: OutputScope, prediction_result: Optional[PredictionResult], output_data,
                         **kwargs):
            file_name = scope.dataset.type.get_file_name(self.file_name)

            with open_writable_text_file(self.output_dir, file_name, fold=scope.fold.index) as text_file:
                text_file.write(output_data.format(self.options, **kwargs))

    class CsvFileSink(Sink):
        """
        Allows to write output data to a CSV file. 
        """

        def __init__(self, output_dir: str, file_name: str, options: Options = Options()):
            """
            :param output_dir:  The path to the directory, where the CSV file should be located
            :param file_name:   The name of the CSV file (without suffix)
            """
            super().__init__(options=options)
            self.output_dir = output_dir
            self.file_name = file_name

        def write_output(self, scope: OutputScope, prediction_result: Optional[PredictionResult], output_data,
                         **kwargs):
            tabular_data = output_data.tabularize(self.options, **kwargs)

            if tabular_data:
                incremental_prediction = prediction_result and not prediction_result.prediction_scope.is_global()

                if incremental_prediction:
                    for row in tabular_data:
                        row['Model size'] = prediction_result.prediction_scope.get_model_size()

                if tabular_data:
                    header = sorted(tabular_data[0].keys())
                    file_name = get_file_name_per_fold(scope.dataset.type.get_file_name(self.file_name), SUFFIX_CSV,
                                                       scope.fold.index)
                    file_path = path.join(self.output_dir, file_name)

                    with open_writable_csv_file(file_path, append=incremental_prediction) as csv_file:
                        csv_writer = create_csv_dict_writer(csv_file, header)

                        for row in tabular_data:
                            csv_writer.writerow(row)

    def __init__(self, sinks: List[Sink]):
        """
        :param sinks: A list that contains all sinks, output data should be written to
        """
        self.sinks = sinks

    @abstractmethod
    def _generate_output_data(self, scope: OutputScope, training_result: Optional[TrainingResult],
                              prediction_result: Optional[PredictionResult]) -> Optional[Any]:
        """
        Must be implemented by subclasses in order to generate the output data that should be written to the available
        sinks.

        :param scope:               The scope of the output data
        :param training_result:     A `TrainingResult` that stores the result of a training process or None, if no model
                                    has been trained
        :param prediction_result:   A `PredictionResult` that stores the result of a prediction process or None, if no
                                    predictions have been obtained
        :return:                    The output data that has been generated or None, if no output data was generated
        """

    def write_output(self,
                     scope: OutputScope,
                     training_result: Optional[TrainingResult] = None,
                     prediction_result: Optional[PredictionResult] = None):
        """
        Generates the output data and writes it to all available sinks.

        :param scope:               The scope of the output data
        :param training_result:     A `TrainingResult` that stores the result of a training process or None, if no model
                                    has been trained
        :param prediction_result:   A `PredictionResult` that stores the result of a prediction process or None, if no
                                    predictions have been obtained
        """
        sinks = self.sinks

        if sinks:
            output_data = self._generate_output_data(scope, training_result, prediction_result)

            if output_data:
                for sink in sinks:
                    sink.write_output(scope, prediction_result, output_data)
