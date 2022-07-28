"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides functions to determine certain characteristics of feature or label matrices.
"""
import numpy as np
from scipy.sparse import issparse


def density(m) -> float:
    """
    Calculates and returns the density of a given feature or label matrix.

    :param m:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_rows, num_cols)`, that stores the feature values
                of training examples or their labels
    :return:    The fraction of non-zero elements in the given matrix among all elements
    """
    num_elements = m.shape[0] * m.shape[1]

    if issparse(m):
        num_non_zero = m.nnz
    else:
        num_non_zero = np.count_nonzero(m)

    return num_non_zero / num_elements if num_elements > 0 else 0


def label_cardinality(y) -> float:
    """
    Calculates and returns the average label cardinality of a given label matrix.

    :param y:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the labels
                of training examples
    :return:    The average number of relevant labels per training example
    """
    if issparse(y):
        y = y.tolil()
        num_relevant_per_example = y.getnnz(axis=1)
    else:
        num_relevant_per_example = np.count_nonzero(y, axis=1)

    return np.average(num_relevant_per_example)


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


def label_imbalance_ratio(y) -> float:
    """
    Calculates and returns the average label imbalance ratio of a given label matrix.

    :param y:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the labels
                of training examples
    :return:    The label imbalance ratio averaged over the available labels
    """
    if issparse(y):
        y = y.tocsc()
        num_relevant_per_label = y.getnnz(axis=0)
    else:
        num_relevant_per_label = np.count_nonzero(y, axis=0)

    num_relevant_per_label = num_relevant_per_label[num_relevant_per_label != 0]

    if num_relevant_per_label.shape[0] > 0:
        return np.average(np.max(num_relevant_per_label) / num_relevant_per_label)
    else:
        return 0.0


class LabelCharacteristics:
    """
    Stores characteristics of a label matrix.
    """

    def __init__(self, y):
        """
        :param y: A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the labels
        """
        self.num_labels = y.shape[1]
        self.label_density = density(y)
        self.avg_label_imbalance_ratio = label_imbalance_ratio(y)
        self.avg_label_cardinality = label_cardinality(y)
        self.num_distinct_label_vectors = distinct_label_vectors(y)
