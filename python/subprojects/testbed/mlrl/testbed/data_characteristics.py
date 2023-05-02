"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing certain characteristics of multi-label data sets. The characteristics can be written to
one or several outputs, e.g., to the console or to a file.
"""
import logging as log
from abc import ABC, abstractmethod
from functools import reduce

from mlrl.common.options import Options
from mlrl.testbed.characteristics import LabelCharacteristics, density, Characteristic, LABEL_CHARACTERISTICS
from mlrl.testbed.data import MetaData, AttributeType
from mlrl.testbed.data_splitting import DataSplit
from mlrl.testbed.format import filter_formattables, format_table, OPTION_DECIMALS, OPTION_PERCENTAGE
from mlrl.testbed.io import open_writable_csv_file, create_csv_dict_writer
from typing import List

OPTION_EXAMPLES = 'examples'

OPTION_FEATURES = 'features'

OPTION_NUMERICAL_FEATURES = 'numerical_features'

OPTION_NOMINAL_FEATURES = 'nominal_features'

OPTION_FEATURE_DENSITY = 'feature_density'

OPTION_FEATURE_SPARSITY = 'feature_sparsity'


class FeatureCharacteristics:
    """
    Stores characteristics of a feature matrix.
    """

    def __init__(self, meta_data: MetaData, x):
        """
        :param meta_data:   The meta-data of the data set
        :param x:           A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that
                            stores the feature values
        """
        self._x = x
        self._meta_data = meta_data
        self.num_examples = x.shape[0]
        self.num_features = x.shape[1]
        self._num_nominal_features = None
        self._feature_density = None

    @property
    def num_nominal_features(self):
        if self._num_nominal_features is None:
            self._num_nominal_features = reduce(
                lambda num, attribute: num + (1 if attribute.attribute_type == AttributeType.NOMINAL else 0),
                self._meta_data.attributes, 0)
        return self._num_nominal_features

    @property
    def num_numerical_features(self):
        return self.num_features - self.num_nominal_features

    @property
    def feature_density(self):
        if self._feature_density is None:
            self._feature_density = density(self._x)
        return self._feature_density

    @property
    def feature_sparsity(self):
        return 1 - self.feature_density


FEATURE_CHARACTERISTICS: List[Characteristic] = [
    Characteristic(OPTION_EXAMPLES, 'Examples', lambda x: x.num_examples),
    Characteristic(OPTION_FEATURES, 'Features', lambda x: x.num_features),
    Characteristic(OPTION_NUMERICAL_FEATURES, 'Numerical Features', lambda x: x.num_nominal_features),
    Characteristic(OPTION_NOMINAL_FEATURES, 'Nominal Features', lambda x: x.num_numerical_features),
    Characteristic(OPTION_FEATURE_DENSITY, 'Feature Density', lambda x: x.feature_density, percentage=True),
    Characteristic(OPTION_FEATURE_SPARSITY, 'Feature Sparsity', lambda x: x.feature_sparsity, percentage=True),
]


class DataCharacteristicsOutput(ABC):
    """
    An abstract base class for all outputs, the characteristics of a data set may be written to.
    """

    @abstractmethod
    def write_data_characteristics(self, data_split: DataSplit, feature_characteristics: FeatureCharacteristics,
                                   label_characteristics: LabelCharacteristics):
        """
        Writes the characteristics of a data set to the output.

        :param data_split:              Information about the split of the available data, the characteristics
                                        correspond to
        :param feature_characteristics: The characteristics of the feature matrix
        :param label_characteristics:   The characteristics of the label matrix
        """
        pass


class DataCharacteristicsLogOutput(DataCharacteristicsOutput):
    """
    Outputs the characteristics of a data set using the logger.
    """

    def __init__(self, options: Options):
        """
        :param options: The options that should be used for writing the characteristics of a data set to the output
        """
        self.feature_characteristic_formattables = filter_formattables(FEATURE_CHARACTERISTICS, [options])
        self.label_characteristic_formattables = filter_formattables(LABEL_CHARACTERISTICS, [options])
        self.percentage = options.get_bool(OPTION_PERCENTAGE, True)
        self.decimals = options.get_int(OPTION_DECIMALS, 2)

    def write_data_characteristics(self, data_split: DataSplit, feature_characteristics: FeatureCharacteristics,
                                   label_characteristics: LabelCharacteristics):
        msg = 'Data characteristics'

        if data_split.is_cross_validation_used():
            msg += ' (Fold ' + str(data_split.get_fold() + 1) + ')'

        msg += ':\n\n%s\n'
        rows = []

        for characteristic in self.feature_characteristic_formattables:
            rows.append([
                characteristic.name,
                characteristic.format(feature_characteristics, percentage=self.percentage, decimals=self.decimals)
            ])

        for characteristic in self.label_characteristic_formattables:
            rows.append([
                characteristic.name,
                characteristic.format(label_characteristics, percentage=self.percentage, decimals=self.decimals)
            ])

        log.info(msg, format_table(rows))


class DataCharacteristicsCsvOutput(DataCharacteristicsOutput):
    """
    Writes the characteristics of a data set to a CSV file.
    """

    def __init__(self, options: Options, output_dir: str):
        """
        :param options:     The options that should be used for writing the characteristics of a data set to the output
        :param output_dir:  The path of the directory, the CSV files should be written to
        """
        self.output_dir = output_dir
        self.feature_characteristic_formattables = filter_formattables(FEATURE_CHARACTERISTICS, [options])
        self.label_characteristic_formattables = filter_formattables(LABEL_CHARACTERISTICS, [options])
        self.percentage = options.get_bool(OPTION_PERCENTAGE, True)
        self.decimals = options.get_int(OPTION_DECIMALS, 0)

    def write_data_characteristics(self, data_split: DataSplit, feature_characteristics: FeatureCharacteristics,
                                   label_characteristics: LabelCharacteristics):
        columns = {}

        for formattable in self.feature_characteristic_formattables:
            columns[formattable] = formattable.format(feature_characteristics,
                                                      percentage=self.percentage,
                                                      decimals=self.decimals)

        for formattable in self.label_characteristic_formattables:
            columns[formattable] = formattable.format(label_characteristics,
                                                      percentage=self.percentage,
                                                      decimals=self.decimals)

        header = sorted(columns.keys())
        with open_writable_csv_file(self.output_dir, 'data_characteristics', data_split.get_fold()) as csv_file:
            csv_writer = create_csv_dict_writer(csv_file, header)
            csv_writer.writerow(columns)


class DataCharacteristicsPrinter:
    """
    A class that allows to print the characteristics of data sets.
    """

    def __init__(self, outputs: List[DataCharacteristicsOutput]):
        """
        :param outputs: The outputs, the characteristics of data sets should be written to
        """
        self.outputs = outputs

    def print(self, meta_data: MetaData, data_split: DataSplit, x, y):
        """
        :param meta_data:   The meta-data of the data set
        :param data_split:  Information about the split of the available data, the characteristics correspond to
        :param x:           A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that
                            stores the feature values
        :param y:           A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores
                            the ground truth labels
        """
        if len(self.outputs) > 0:
            feature_characteristics = FeatureCharacteristics(meta_data, x)
            label_characteristics = LabelCharacteristics(y)

            for output in self.outputs:
                output.write_data_characteristics(data_split, feature_characteristics, label_characteristics)
