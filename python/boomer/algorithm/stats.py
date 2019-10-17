#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides utility functions and classes to obtain statistics about multi-label data sets.
"""
import numpy as np


def get_num_examples(x: np.ndarray) -> int:
    """
    Returns the number of examples contained in a data set.

    :param x:   An array of dtype float, shape `(num_examples, num_features)`, representing the features of examples
    :return:    The number of examples contained in the given data set
    """
    return np.shape(x)[0]


def get_num_features(x: np.ndarray) -> int:
    """
    Returns the number of features contained in a data set.

    :param x:   An array of dtype float, shape `(num_examples, num_features)`, representing the features of examples
    :return:    The number of features contained in the given data set
    """
    return np.shape(x)[1]


def get_num_labels(y: np.ndarray) -> int:
    """
    Returns the number of labels contained in a label set.

    :param y:   An array of dtype float, shape `(num_examples, num_labels)`, representing the labels of examples
    :return:    The number of labels contained in the given label set
    """
    return np.shape(y)[1]


class Stats:
    """
    Stores useful information about a multi-label training data set, such as the number of examples, features, and
    labels.
    """

    def __init__(self, num_examples: int, num_features: int, num_labels: int):
        """
        :param num_examples:    The number of examples contained in the training data set
        :param num_features:    The number of features contained in the training data set
        :param num_labels:      The number of labels contained in the training data set
        """
        self.num_examples = num_examples
        self.num_features = num_features
        self.num_labels = num_labels

    @staticmethod
    def create_stats(x: np.ndarray, y: np.ndarray) -> 'Stats':
        """
        Creates 'Stats' storing information about a specific multi-label training data set.

        :param x:   An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                    training examples
        :param y:   An array of dtype float, shape `(num_examples, num_labels)`, representing the labels of the training
                    examples
        :return:    'Stats' storing information about the given data set
        """
        return Stats(num_examples=get_num_examples(x), num_features=get_num_features(x), num_labels=get_num_labels(y))
