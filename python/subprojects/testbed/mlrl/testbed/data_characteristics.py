"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for printing certain characteristics of multi-label data sets. The characteristics can be written to
one or several outputs, e.g., to the console or to a file.
"""
import logging as log
from abc import ABC, abstractmethod
from functools import reduce

from mlrl.testbed.characteristics import LabelCharacteristics, density
from mlrl.testbed.data import MetaData, AttributeType
from mlrl.testbed.data_splitting import DataSplit
from mlrl.testbed.format import format_table
from mlrl.testbed.io import open_writable_csv_file, create_csv_dict_writer
from typing import List

COLUMN_EXAMPLES = 'Examples'

COLUMN_FEATURES = 'Features'

COLUMN_NUMERICAL_FEATURES = 'Numerical features'

COLUMN_NOMINAL_FEATURES = 'Nominal features'

COLUMN_FEATURE_DENSITY = 'Feature density'

COLUMN_FEATURE_SPARSITY = 'Feature sparsity'

COLUMN_LABELS = 'Labels'

COLUMN_LABEL_DENSITY = 'Label density'

COLUMN_LABEL_SPARSITY = 'Label sparsity'

COLUMN_LABEL_IMBALANCE_RATIO = 'Label imbalance ratio'

COLUMN_LABEL_CARDINALITY = 'Label cardinality'

COLUMN_DISTINCT_LABEL_VECTORS = 'Distinct label vectors'


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

    def write_data_characteristics(self, data_split: DataSplit, feature_characteristics: FeatureCharacteristics,
                                   label_characteristics: LabelCharacteristics):
        msg = 'Data characteristics'

        if data_split.is_cross_validation_used():
            msg += ' (Fold ' + str(data_split.get_fold() + 1) + ')'

        msg += ':\n\n%s\n'
        rows = [
            [COLUMN_EXAMPLES, str(feature_characteristics.num_examples)],
            [COLUMN_FEATURES, str(feature_characteristics.num_features)],
            [COLUMN_NUMERICAL_FEATURES, str(feature_characteristics.num_numerical_features)],
            [COLUMN_NOMINAL_FEATURES, str(feature_characteristics.num_nominal_features)],
            [COLUMN_FEATURE_DENSITY, str(feature_characteristics.feature_density)],
            [COLUMN_FEATURE_SPARSITY, str(1 - feature_characteristics.feature_density)],
            [COLUMN_LABELS, str(label_characteristics.num_labels)],
            [COLUMN_LABEL_DENSITY, str(label_characteristics.label_density)],
            [COLUMN_LABEL_SPARSITY, str(label_characteristics.label_sparsity)],
            [COLUMN_LABEL_IMBALANCE_RATIO, str(label_characteristics.avg_label_imbalance_ratio)],
            [COLUMN_LABEL_CARDINALITY, str(label_characteristics.avg_label_cardinality)],
            [COLUMN_DISTINCT_LABEL_VECTORS, str(label_characteristics.num_distinct_label_vectors)]
        ]
        log.info(msg, format_table(rows))


class DataCharacteristicsCsvOutput(DataCharacteristicsOutput):
    """
    Writes the characteristics of a data set to a CSV file.
    """

    def __init__(self, output_dir: str):
        """
        :param output_dir: The path of the directory, the CSV files should be written to
        """
        self.output_dir = output_dir

    def write_data_characteristics(self, data_split: DataSplit, feature_characteristics: FeatureCharacteristics,
                                   label_characteristics: LabelCharacteristics):
        columns = {
            COLUMN_EXAMPLES: feature_characteristics.num_examples,
            COLUMN_FEATURES: feature_characteristics.num_features,
            COLUMN_NUMERICAL_FEATURES: feature_characteristics.num_numerical_features,
            COLUMN_NOMINAL_FEATURES: feature_characteristics.num_nominal_features,
            COLUMN_FEATURE_DENSITY: feature_characteristics.feature_density,
            COLUMN_FEATURE_SPARSITY: 1 - feature_characteristics.feature_density,
            COLUMN_LABELS: label_characteristics.num_labels,
            COLUMN_LABEL_DENSITY: label_characteristics.label_density,
            COLUMN_LABEL_SPARSITY: label_characteristics.label_sparsity,
            COLUMN_LABEL_IMBALANCE_RATIO: label_characteristics.avg_label_imbalance_ratio,
            COLUMN_LABEL_CARDINALITY: label_characteristics.avg_label_cardinality,
            COLUMN_DISTINCT_LABEL_VECTORS: label_characteristics.num_distinct_label_vectors
        }
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
