"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing certain characteristics of binary predictions. The characteristics can be written to one
or several outputs, e.g., to the console or to a file.
"""
import logging as log
from abc import ABC, abstractmethod

from mlrl.common.options import Options
from mlrl.testbed.characteristics import LabelCharacteristics, LABEL_CHARACTERISTICS
from mlrl.testbed.data_splitting import DataSplit, DataType
from mlrl.testbed.format import filter_formattables, format_table, OPTION_PERCENTAGE, OPTION_DECIMALS
from mlrl.testbed.io import open_writable_csv_file, create_csv_dict_writer
from mlrl.testbed.predictions import PredictionScope
from typing import List


class PredictionCharacteristicsOutput(ABC):
    """
    An abstract base class for all outputs, the characteristics of binary predictions may be written to.
    """

    @abstractmethod
    def write_prediction_characteristics(self, data_split: DataSplit, data_type: DataType,
                                         prediction_scope: PredictionScope, characteristics: LabelCharacteristics):
        """
        Writes the characteristics of a data set to the output.

        :param data_split:          The split of the available data, the characteristics correspond to
        :param data_type:           Specifies whether the predictions correspond to the training or test data
        :param prediction_scope:    Specifies whether the predictions have been obtained from a global model or
                                    incrementally
        :param characteristics:     The characteristics of the predictions
        """
        pass


class PredictionCharacteristicsLogOutput(PredictionCharacteristicsOutput):
    """
    Outputs the characteristics of binary predictions using the logger.
    """

    def __init__(self, options: Options):
        """
        :param options: The options that should be used for writing the characteristics of predictions to the output
        """
        self.formattables = filter_formattables(LABEL_CHARACTERISTICS, [options])
        self.percentage = options.get_bool(OPTION_PERCENTAGE, True)
        self.decimals = options.get_int(OPTION_DECIMALS, 2)

    def write_prediction_characteristics(self, data_split: DataSplit, data_type: DataType,
                                         prediction_scope: PredictionScope, characteristics: LabelCharacteristics):
        msg = 'Prediction characteristics for ' + data_type.value + ' data'

        if not prediction_scope.is_global():
            msg += ' using a model of size ' + str(prediction_scope.get_model_size())

        if data_split.is_cross_validation_used():
            msg += ' (Fold ' + str(data_split.get_fold() + 1) + ')'

        msg += ':\n\n%s\n'
        rows = []

        for formattable in self.formattables:
            rows.append([
                formattable.name,
                formattable.format(characteristics, percentage=self.percentage, decimals=self.decimals)
            ])

        log.info(msg, format_table(rows))


class PredictionCharacteristicsCsvOutput(PredictionCharacteristicsOutput):
    """
    Writes the characteristics of binary predictions to a CSV file.
    """

    COLUMN_MODEL_SIZE = 'Model size'

    def __init__(self, options: Options, output_dir: str):
        """
        :param options:     The options that should be used for writing the characteristics of predictions to the output
        :param output_dir:  The path of the directory, the CSV files should be written to
        """
        self.output_dir = output_dir
        self.formattables = filter_formattables(LABEL_CHARACTERISTICS, [options])
        self.percentage = options.get_bool(OPTION_PERCENTAGE, True)
        self.decimals = options.get_int(OPTION_DECIMALS, 0)

    def write_prediction_characteristics(self, data_split: DataSplit, data_type: DataType,
                                         prediction_scope: PredictionScope, characteristics: LabelCharacteristics):
        columns = {}

        for formattable in self.formattables:
            columns[formattable] = formattable.format(characteristics,
                                                      percentage=self.percentage,
                                                      decimals=self.decimals)

        header = sorted(columns.keys())
        incremental_prediction = not prediction_scope.is_global()

        if incremental_prediction:
            columns[PredictionCharacteristicsCsvOutput.COLUMN_MODEL_SIZE] = prediction_scope.get_model_size()
            header = [PredictionCharacteristicsCsvOutput.COLUMN_MODEL_SIZE] + header

        with open_writable_csv_file(self.output_dir,
                                    data_type.get_file_name('prediction_characteristics'),
                                    data_split.get_fold(),
                                    append=incremental_prediction) as csv_file:
            csv_writer = create_csv_dict_writer(csv_file, header)
            csv_writer.writerow(columns)


class PredictionCharacteristicsPrinter:
    """
    A class that allows to print the characteristics of binary predictions.
    """

    def __init__(self, outputs: List[PredictionCharacteristicsOutput]):
        """
        :param outputs: The outputs, the characteristics of binary predictions should be written to
        """
        self.outputs = outputs

    def print(self, data_split: DataSplit, data_type: DataType, prediction_scope: PredictionScope, y):
        """
        :param data_split:          The split of the available data, the characteristics correspond to
        :param data_type:           Specifies whether the predictions correspond to the training or test data
        :param prediction_scope:    Specifies whether the predictions have been obtained from a global model or
                                    incrementally
        :param y:                   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that
                                    stores the predictions
        """
        if len(self.outputs) > 0:
            characteristics = LabelCharacteristics(y)

            for output in self.outputs:
                output.write_prediction_characteristics(data_split, data_type, prediction_scope, characteristics)
