"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for writing output data to sinks like the console or output files.
"""
import logging as log

from abc import ABC, abstractmethod
from os import path
from typing import Any, Dict, List, Optional

from mlrl.common.config.options import Options

from mlrl.testbed.data import MetaData
from mlrl.testbed.data_splitting import DataSplit, DataType
from mlrl.testbed.io import SUFFIX_CSV, create_csv_dict_writer, get_file_name_per_fold, open_writable_csv_file, \
    open_writable_text_file
from mlrl.testbed.prediction_scope import PredictionScope, PredictionType
from mlrl.testbed.problem_type import ProblemType


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
        def write_output(self, problem_type: ProblemType, meta_data: MetaData, data_split: DataSplit,
                         data_type: Optional[DataType], prediction_scope: Optional[PredictionScope], output_data,
                         **kwargs):
            """
            Must be implemented by subclasses in order to write output data to the sink.

            :param problem_type:        The type of the machine learning problem
            :param meta_data:           The meta-data of the data set
            :param data_split:          Information about the split of the available data, the output data corresponds
                                        to
            :param data_type:           Specifies whether the predictions and ground truth correspond to the training or
                                        test data or None, if no predictions have been obtained
            :param prediction_scope:    Specifies whether the predictions have been obtained from a global model or
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

        def write_output(self, problem_type: ProblemType, meta_data: MetaData, data_split: DataSplit,
                         data_type: Optional[DataType], prediction_scope: Optional[PredictionScope], output_data,
                         **kwargs):
            message = self.title

            if data_type:
                message += ' for ' + data_type.value + ' data'

            if prediction_scope and not prediction_scope.is_global():
                message += ' using a model of size ' + str(prediction_scope.get_model_size())

            if data_split.is_cross_validation_used():
                fold = data_split.get_fold()
                message += ' ('
                message += 'Average across ' + str(
                    data_split.get_num_folds()) + ' folds' if fold is None else 'Fold ' + str(fold + 1)
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

        def write_output(self, problem_type: ProblemType, meta_data: MetaData, data_split: DataSplit,
                         data_type: Optional[DataType], prediction_scope: Optional[PredictionScope], output_data,
                         **kwargs):
            file_name = data_type.get_file_name(self.file_name) if data_type else self.file_name

            with open_writable_text_file(directory=self.output_dir, file_name=file_name,
                                         fold=data_split.get_fold()) as text_file:
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

        def write_output(self, problem_type: ProblemType, meta_data: MetaData, data_split: DataSplit,
                         data_type: Optional[DataType], prediction_scope: Optional[PredictionScope], output_data,
                         **kwargs):
            tabular_data = output_data.tabularize(self.options, **kwargs)

            if tabular_data:
                incremental_prediction = prediction_scope and not prediction_scope.is_global()

                if incremental_prediction:
                    for row in tabular_data:
                        row['Model size'] = prediction_scope.get_model_size()

                if tabular_data:
                    header = sorted(tabular_data[0].keys())
                    file_name = data_type.get_file_name(self.file_name) if data_type else self.file_name
                    file_name = get_file_name_per_fold(file_name, SUFFIX_CSV, data_split.get_fold())
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
    def _generate_output_data(self, problem_type: ProblemType, meta_data: MetaData, x, y, data_split: DataSplit,
                              learner, data_type: Optional[DataType], prediction_type: Optional[PredictionType],
                              prediction_scope: Optional[PredictionScope], predictions: Optional[Any],
                              train_time: float, predict_time: float) -> Optional[Any]:
        """
        Must be implemented by subclasses in order to generate the output data that should be written to the available
        sinks.

        :param meta_data:           The meta-data of the data set
        :param x:                   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                    `(num_examples, num_features)`, that stores the feature values
        :param y:                   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                    `(num_examples, num_outputs)`, that stores the ground truth
        :param data_split:          Information about the split of the available data, the output data corresponds to
        :param learner:             The learner that has been trained
        :param data_type:           Specifies whether the predictions and ground truth correspond to the training or
                                    test data or None, if no predictions have been obtained
        :param prediction_type:     The type of the predictions or None, if no predictions have been obtained
        :param prediction_scope:    Specifies whether the predictions have been obtained from a global model or
                                    incrementally or None, if no predictions have been obtained
        :param predictions:         A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                    `(num_examples, num_outputs)`, that stores the predictions for the query examples or
                                    None, if no predictions have been obtained
        :param train_time:          The time needed for training or 0, if no model has been trained
        :param predict_time:        The time needed for prediction or 0, if no predictions have been obtained
        :return:                    The output data that has been generated or None, if no output data was generated
        """

    def write_output(self,
                     problem_type: ProblemType,
                     meta_data: MetaData,
                     x,
                     y,
                     data_split: DataSplit,
                     learner,
                     data_type: Optional[DataType] = None,
                     prediction_type: Optional[PredictionType] = None,
                     prediction_scope: Optional[PredictionScope] = None,
                     predictions: Optional[Any] = None,
                     train_time: float = 0,
                     predict_time: float = 0):
        """
        Generates the output data and writes it to all available sinks.

        :param problem_type:        The type of the machine learning problem
        :param meta_data:           The meta-data of the data set
        :param x:                   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                    `(num_examples, num_features)`, that stores the feature values
        :param y:                   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                    `(num_examples, num_outputs)`, that stores the ground truth
        :param data_split:          Information about the split of the available data, the output data corresponds to
        :param learner:             The learner that has been trained
        :param data_type:           Specifies whether the predictions and ground truth correspond to the training or
                                    test data or None, if no predictions have been obtained
        :param prediction_type:     The type of the predictions or None, if no predictions have been obtained
        :param prediction_scope:    Specifies whether the predictions have been obtained from a global model or
                                    incrementally or None, if no predictions have been obtained
        :param predictions:         A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                    `(num_examples, num_outputs)`, that stores the predictions for the query examples or
                                    None, if no predictions have been obtained
        :param train_time:          The time needed for training or 0, if no model has been trained
        :param predict_time:        The time needed for prediction or 0, if no predictions have been obtained
        """
        sinks = self.sinks

        if sinks:
            output_data = self._generate_output_data(problem_type, meta_data, x, y, data_split, learner, data_type,
                                                     prediction_type, prediction_scope, predictions, train_time,
                                                     predict_time)

            if output_data:
                for sink in sinks:
                    sink.write_output(problem_type, meta_data, data_split, data_type, prediction_scope, output_data)
