"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing certain characteristics of multi-label data sets. The characteristics can be written to
one or several outputs, e.g., to the console or to a file.
"""
import logging as log
from abc import ABC, abstractmethod
from functools import reduce
from typing import List

from mlrl.testbed.characteristics import LabelCharacteristics, density
from mlrl.testbed.data import MetaData, AttributeType
from mlrl.testbed.io import open_writable_csv_file, create_csv_dict_writer
from mlrl.testbed.training import DataPartition


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
        self.num_examples = x.shape[0]
        self.num_features = x.shape[1]
        self.num_nominal_features = reduce(
            lambda num, attribute: num + (1 if attribute.attribute_type == AttributeType.NOMINAL else 0),
            meta_data.attributes, 0)
        self.num_numerical_features = self.num_features - self.num_nominal_features
        self.feature_density = density(x)


class DataCharacteristicsOutput(ABC):
    """
    An abstract base class for all outputs, the characteristics of a data set may be written to.
    """

    @abstractmethod
    def write_data_characteristics(self, data_partition: DataPartition, feature_characteristics: FeatureCharacteristics,
                                   label_characteristics: LabelCharacteristics):
        """
        Writes the characteristics of a data set to the output.

        :param data_partition:          Information about the partition of data, the characteristics correspond to
        :param feature_characteristics: The characteristics of the feature matrix
        :param label_characteristics:   The characteristics of the label matrix
        """
        pass


class DataCharacteristicsLogOutput(DataCharacteristicsOutput):
    """
    Outputs the characteristics of a data set using the logger.
    """

    def write_data_characteristics(self, data_partition: DataPartition, feature_characteristics: FeatureCharacteristics,
                                   label_characteristics: LabelCharacteristics):
        msg = 'Data characteristics'

        if data_partition.is_cross_validation_used():
            msg += ' (Fold ' + str(data_partition.get_fold() + 1) + ')'

        msg += ':\n\n'
        msg += 'Examples: ' + str(feature_characteristics.num_examples) + '\n'
        msg += 'Features: ' + str(feature_characteristics.num_features) + ' (' + str(
            feature_characteristics.num_numerical_features) + ' numerical, ' + str(
            feature_characteristics.num_nominal_features) + ' nominal)\n'
        msg += 'Feature density: ' + str(feature_characteristics.feature_density) + '\n'
        msg += 'Feature sparsity: ' + str(1 - feature_characteristics.feature_density) + '\n'
        msg += 'Labels: ' + str(label_characteristics.num_labels) + '\n'
        msg += 'Label density: ' + str(label_characteristics.label_density) + '\n'
        msg += 'Label sparsity: ' + str(1 - label_characteristics.label_density) + '\n'
        msg += 'Label imbalance ratio: ' + str(label_characteristics.avg_label_imbalance_ratio) + '\n'
        msg += 'Label cardinality: ' + str(label_characteristics.avg_label_cardinality) + '\n'
        msg += 'Distinct label vectors: ' + str(label_characteristics.num_distinct_label_vectors) + '\n'
        log.info(msg)


class DataCharacteristicsCsvOutput(DataCharacteristicsOutput):
    """
    Writes the characteristics of a data set to a CSV file.
    """

    def __init__(self, output_dir: str):
        """
        :param output_dir: The path of the directory, the CSV files should be written to
        """
        self.output_dir = output_dir

    def write_data_characteristics(self, data_partition: DataPartition, feature_characteristics: FeatureCharacteristics,
                                   label_characteristics: LabelCharacteristics):
        columns = {
            'Examples': feature_characteristics.num_examples,
            'Features': feature_characteristics.num_features,
            'Numerical features': feature_characteristics.num_numerical_features,
            'Nominal features': feature_characteristics.num_nominal_features,
            'Feature density': feature_characteristics.feature_density,
            'Feature sparsity': 1 - feature_characteristics.feature_density,
            'Labels': label_characteristics.num_labels,
            'Label density': label_characteristics.label_density,
            'Label sparsity': 1 - label_characteristics.label_density,
            'Label imbalance ratio': label_characteristics.avg_label_imbalance_ratio,
            'Label cardinality': label_characteristics.avg_label_cardinality,
            'Distinct label vectors': label_characteristics.num_distinct_label_vectors
        }
        header = sorted(columns.keys())
        with open_writable_csv_file(self.output_dir, 'data_characteristics', data_partition.get_fold()) as csv_file:
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

    def print(self, meta_data: MetaData, data_partition: DataPartition, x, y):
        """
        :param meta_data:       The meta-data of the data set
        :param data_partition:  Information about the partition of data, the characteristics correspond to
        :param x:               A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that
                                stores the feature values
        :param y:               A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that
                                stores the ground truth labels
        """
        if len(self.outputs) > 0:
            feature_characteristics = FeatureCharacteristics(meta_data, x)
            label_characteristics = LabelCharacteristics(y)

            for output in self.outputs:
                output.write_data_characteristics(data_partition, feature_characteristics, label_characteristics)
