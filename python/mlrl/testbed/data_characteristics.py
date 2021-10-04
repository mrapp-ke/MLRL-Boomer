#!/usr/bin/python

"""
Author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides functions to determine certain characteristics of multi-label data sets.
"""
import logging as log
from abc import ABC, abstractmethod
from functools import reduce
from typing import List

import numpy as np
from scipy.sparse import issparse

from mlrl.testbed.data import MetaData, AttributeType


def density(m) -> float:
    """
    Calculates and returns the density of a given feature or label matrix.

    :param m:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_rows, num_cols)`, that stores the feature values
                of training examples or their labels
    :return:    The fraction of non-zero elements in the given matrix among all elements
    """
    num_elements = m.shape[0] * m.shape[1]

    if issparse(m):
        return m.nnz / num_elements
    else:
        return np.count_nonzero(m) / num_elements


def label_cardinality(y) -> float:
    """
    Calculates and returns the label cardinality of a given label matrix.

    :param y:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the labels
                of training examples
    :return:    The average number of relevant labels per training example
    """
    if issparse(y):
        y = y.tolil()
        cardinality = 0.0

        for i in range(y.shape[0]):
            row = y.getrowview(i)
            cardinality += ((row.nnz - cardinality) / (i + 1))

        return cardinality
    else:
        return np.average(np.count_nonzero(y, axis=1))


def distinct_label_vectors(y) -> int:
    """
    Determines and returns the number of distinct label vectors in a label matrix.

    :param y:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the labels
                of training examples
    :return:    The number of distinct label vectors in the given matrix
    """
    if issparse(y):
        y = y.tolil()
        return np.unique(y.rows).shape[0]
    else:
        return np.unique(y, axis=0).shape[0]


class DataCharacteristics:
    """
    Stores characteristics of a multi-label data set.
    """

    def __init__(self, num_examples: int, num_nominal_features: int, num_numerical_features: int,
                 feature_density: float, num_labels: int, label_density: float, avg_label_cardinality: float,
                 num_distinct_label_vectors: int):
        """
        :param num_examples:                The number of examples in the data set
        :param num_nominal_features:        The number of nominal features in the data set
        :param num_numerical_features:      The number of numerical features in the data set
        :param feature_density:             The feature density
        :param num_labels:                  The number of labels in the data set
        :param label_density:               The label density
        :param avg_label_cardinality:       The average label cardinality
        :param num_distinct_label_vectors:  The number of distinct label vectors in the data set
        """
        self.num_examples = num_examples
        self.num_nominal_features = num_nominal_features
        self.num_numerical_features = num_numerical_features
        self.feature_density = feature_density
        self.num_labels = num_labels
        self.label_density = label_density
        self.avg_label_cardinality = avg_label_cardinality
        self.num_distinct_label_vectors = num_distinct_label_vectors


class DataCharacteristicsOutput(ABC):
    """
    An abstract base class for all outputs, the characteristics of a data set may be written to.
    """

    @abstractmethod
    def write_data_characteristics(self, experiment_name: str, characteristics: DataCharacteristics, total_folds: int,
                                   fold: int = None):
        """
        Writes the characteristics of a data set to the output.

        :param experiment_name: The name of the experiment
        :param characteristics: The characteristics of the data set
        :param total_folds:     The total number of folds
        :param fold:            The fold for which the characteristics should be written or None, if no cross validation
                                is used
        """
        pass


class DataCharacteristicsLogOutput(DataCharacteristicsOutput):
    """
    Outputs the characteristics of a data set using the logger.
    """

    def write_data_characteristics(self, experiment_name: str, characteristics: DataCharacteristics, total_folds: int,
                                   fold: int = None):
        msg = 'Data characteristics for experiment \"' + experiment_name + '\"' + (
            ' (Fold ' + str(fold + 1) + ')' if fold is not None else '') + ':\n\n'
        msg += 'Examples: ' + str(characteristics.num_examples) + '\n'
        msg += 'Features: ' + str(
            characteristics.num_nominal_features + characteristics.num_numerical_features) + ' (' + str(
            characteristics.num_numerical_features) + ' numerical, ' + str(
            characteristics.num_nominal_features) + ' nominal)\n'
        msg += 'Feature density: ' + str(characteristics.feature_density) + '\n'
        msg += 'Feature sparsity: ' + str(1 - characteristics.feature_density) + '\n'
        msg += 'Labels: ' + str(characteristics.num_labels) + '\n'
        msg += 'Label density: ' + str(characteristics.label_density) + '\n'
        msg += 'Label sparsity: ' + str(1 - characteristics.label_density) + '\n'
        msg += 'Label cardinality: ' + str(characteristics.avg_label_cardinality) + '\n'
        msg += 'Distinct label vectors: ' + str(characteristics.num_distinct_label_vectors) + '\n'
        log.info(msg)


class DataCharacteristicsPrinter(ABC):
    """
    A class that allows to print the characteristics of data sets.
    """

    def __init__(self, outputs: List[DataCharacteristicsOutput]):
        """
        :param outputs: The outputs, the characteristics of data sets should be written to
        """
        self.outputs = outputs

    def print(self, experiment_name: str, x, y, meta_data: MetaData, current_fold: int, num_folds: int):
        """
        :param experiment_name: The name of the experiment
        :param x:               A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that
                                stores the feature values
        :param y:               A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that
                                stores the ground truth labels
        :param meta_data:       The meta data of the data set
        :param current_fold:    The current fold
        :param num_folds:       The total number of folds
        """
        if len(self.outputs) > 0:
            num_examples = x.shape[0]
            num_features = x.shape[1]
            num_nominal_features = reduce(
                lambda num, attribute: num + (1 if attribute.attribute_type == AttributeType.NOMINAL else 0),
                meta_data.attributes, 0)
            num_numerical_features = num_features - num_nominal_features
            feature_density = density(x)
            num_labels = y.shape[1]
            label_density = density(y)
            avg_label_cardinality = label_cardinality(y)
            num_distinct_label_vectors = distinct_label_vectors(y)
            characteristics = DataCharacteristics(num_examples=num_examples, num_nominal_features=num_nominal_features,
                                                  num_numerical_features=num_numerical_features,
                                                  feature_density=feature_density, num_labels=num_labels,
                                                  label_density=label_density,
                                                  avg_label_cardinality=avg_label_cardinality,
                                                  num_distinct_label_vectors=num_distinct_label_vectors)
            for output in self.outputs:
                output.write_data_characteristics(experiment_name, characteristics, num_folds,
                                                  current_fold if num_folds > 1 else None)
