#!/usr/bin/python

"""
Author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides functions to determine certain characteristics of multi-label datasets.
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
        return m.nnz / num_elements
    else:
        return np.count_nonzero(m) / num_elements


def label_cardinality(m) -> float:
    """
    Calculates and returns the label cardinality of a given label matrix.

    :param m:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the labels
                of training examples
    :return:    The average number of relevant labels per training example
    """
    if issparse(m):
        m = m.tolil()
        cardinality = 0.0

        for i in range(m.shape[0]):
            row = m.getrowview(i)
            cardinality += ((row.nnz - cardinality) / (i + 1))

        return cardinality
    else:
        return np.average(np.count_nonzero(m, axis=1))


def num_distinct_label_vectors(m) -> int:
    """
    Determines and returns the number of distinct label vectors in a label matrix.

    :param m:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the labels
                of training examples
    :return:    The number of distinct label vectors in the given matrix
    """
    if issparse(m):
        m = m.tolil()
        return np.unique(m.rows).shape[0]
    else:
        return np.unique(m, axis=0).shape[0]
