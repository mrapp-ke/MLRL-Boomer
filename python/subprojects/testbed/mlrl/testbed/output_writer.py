"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for writing output data to sinks like the console or output files.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
import logging as log

from mlrl.testbed.data import MetaData
from mlrl.testbed.data_splitting import DataSplit, DataType
from mlrl.testbed.io import open_writable_txt_file, open_writable_csv_file, create_csv_dict_writer
from mlrl.testbed.predictions import PredictionType, PredictionScope
from mlrl.common.options import Options


class Formattable(ABC):
    """
    An abstract base class for all classes from which a textual representation can be created.
    """

    @abstractmethod
    def format(self, options: Options) -> str:
        """
        Creates and returns a textual representation of the object.

        :param options: Options to be taken into account
        :return:        The textual representation that has been created
        """
        pass


class Tabularizable(ABC):
    """
    An abstract base class for all classes from which a tabular representation can be created.
    """

    def tabularize(self, options: Options) -> List[Dict[str, str]]:
        """
        Creates and returns a tabular representation of the object.

        :param options: Options to be taken into account
        :return:        The tabular representation that has been created
        """
        pass


class OutputWriter(ABC):
    """
    An abstract base class for all classes that allow to write output data to one or several sinks, e.g., the console or
    output files.
    """

    class Sink(ABC):
        """
        An abstract base class for all sinks, output data may be written to.
        """

        @abstractmethod
        def write_output(self, data_split: DataSplit, data_type: Optional[DataType],
                         prediction_scope: Optional[PredictionScope], output_data):
            """
            Must be implemented by subclasses in order to write output data to the sink.

            :param data_split:          Information about the split of the available data, the output data corresponds
                                        to
            :param data_type:           Specifies whether the predictions and ground truth labels correspond to the
                                        training or test data or None, if no predictions have been obtained
            :param prediction_scope:    Specifies whether the predictions have been obtained from a global model or
                                        incrementally or None, if no predictions have been obtained
            :param output_data:         The output data that should be written to the sink
            """
            pass

    class LogSink(Sink):
        """
        Allows to write output data to the console.
        """

        def __init__(self, title: str, options: Options = Options()):
            """
            :param title:   A title that is printed before the actual output data
            :param options: Options to be taken into account
            """
            self.title = title
            self.options = options

        def write_output(self, data_split: DataSplit, data_type: Optional[DataType],
                         prediction_scope: Optional[PredictionScope], output_data):
            message = self.title

            if data_type is not None:
                message += ' for ' + data_type.value + ' data'

            if prediction_scope is not None and not prediction_scope.is_global():
                message += ' using a model of size ' + str(prediction_scope.get_model_size())

            if data_split.is_cross_validation_used():
                message += ' (Fold ' + str(data_split.get_fold() + 1) + ')'

            message += ':\n\n' + output_data.format(self.options) + '\n'
            log.info(message)

    class TxtSink(Sink):
        """
        Allows to write output data to a text file.
        """

        def __init__(self, output_dir: str, file_name: str, options: Options = Options()):
            """
            :param output_dir:  The path of the directory, where the text file should be located
            :param file_name:   The name of the text file (without suffix)
            :param options:     Options to be taken into account
            """
            self.output_dir = output_dir
            self.file_name = file_name
            self.options = options

        def write_output(self, data_split: DataSplit, data_type: Optional[DataType],
                         prediction_scope: Optional[PredictionScope], output_data):
            with open_writable_txt_file(self.output_dir, self.file_name, data_split.get_fold()) as txt_file:
                txt_file.write(output_data.format(self.options))

    class CsvSink(Sink):
        """
        Allows to write output data to a CSV file. 
        """

        def __init__(self, output_dir: str, file_name: str, options: Options = Options()):
            """
            :param output_dir:  The path of the directory, where the text file should be located
            :param file_name:   The name of the text file (without suffix)
            :param options:     Options to be taken into account
            """
            self.output_dir = output_dir
            self.file_name = file_name
            self.options = options

        def write_output(self, data_split: DataSplit, data_type: Optional[DataType],
                         prediction_scope: Optional[PredictionScope], output_data):
            tabular_data = output_data.tabularize(self.options)
            incremental_prediction = prediction_scope is not None and not prediction_scope.is_global()

            if incremental_prediction:
                for row in tabular_data:
                    row['Model size'] = prediction_scope.get_model_size()

            if len(tabular_data) > 0:
                header = sorted(tabular_data[0].keys())

                with open_writable_csv_file(directory=self.output_dir,
                                            file_name=self.file_name,
                                            fold=data_split.get_fold(),
                                            append=incremental_prediction) as csv_file:
                    csv_writer = create_csv_dict_writer(csv_file, header)

                    for row in tabular_data:
                        csv_writer.writerow(row)

    def __init__(self, sinks: List[Sink]):
        """
        :param sinks: A list that contains all sinks, output data should be written to
        """
        self.sinks = sinks

    @abstractmethod
    def _generate_output_data(self, meta_data: MetaData, x, y, data_split: DataSplit, learner,
                              prediction_type: Optional[PredictionType], predictions: Optional[Any]) -> Optional[Any]:
        """
        Must be implemented by subclasses in order to generate the output data that should be written to the available
        sinks.

        :param meta_data:       The meta-data of the data set
        :param x:               A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that
                                stores the feature values
        :param y:               A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that
                                stores the ground truth labels
        :param data_split:      Information about the split of the available data, the output data corresponds to
        :param learner:         The learner that has been trained
        :param prediction_type: The type of the predictions or None, if no predictions have been obtained
        :param predictions:     A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that
                                stores the predictions for the query examples or None, if no predictions have been
                                obtained
        :return:                The output data that has been generated or None, if no output data was generated
        """
        pass

    def write_output(self,
                     meta_data: MetaData,
                     x,
                     y,
                     data_split: DataSplit,
                     learner,
                     data_type: Optional[DataType] = None,
                     prediction_type: Optional[PredictionType] = None,
                     prediction_scope: Optional[PredictionScope] = None,
                     predictions: Optional[Any] = None):
        """
        Generates the output data and writes it to all available sinks.

        :param meta_data:           The meta-data of the data set
        :param x:                   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`,
                                    that stores the feature values
        :param y:                   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that
                                    stores the ground truth labels
        :param data_split:          Information about the split of the available data, the output data corresponds to
        :param learner:             The learner that has been trained
        :param data_type:           Specifies whether the predictions and ground truth labels correspond to the training
                                    or test data or None, if no predictions have been obtained
        :param prediction_type:     The type of the predictions or None, if no predictions have been obtained
        :param prediction_scope:    Specifies whether the predictions have been obtained from a global model or
                                    incrementally or None, if no predictions have been obtained
        :param predictions:         A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that
                                    stores the predictions for the query examples or None, if no predictions have been
                                    obtained
        """
        sinks = self.sinks

        if len(sinks) > 0:
            output_data = self._generate_output_data(meta_data, x, y, data_split, learner, prediction_type, predictions)

            if output_data is not None:
                for sink in sinks:
                    sink.write_output(data_split, data_type, prediction_scope, output_data)
