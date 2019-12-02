#!/usr/bin/python


"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for implementing multi-label classifiers or rankers.
"""
from abc import ABC, abstractmethod

import numpy as np
from skmultilearn.base import MLClassifierBase


class Randomized(ABC):
    """
    A base class for all classifiers, rankers or modules that use RNGs.

    Attributes
        random_state   The seed to be used by RNGs
    """

    random_state: int = 1


class Module(Randomized):
    """
    A base class for all modules, a multi-label classifier or ranker consists of.
    """


class MLLearner(MLClassifierBase, Randomized):
    """
    A base class for all multi-label classifiers or rankers.

    Attributes
        fold    The current fold or None, if no cross validation is used
    """

    fold: int = None

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> 'MLLearner':
        """
        Trains the classifier or ranker on the given training data.

        :param x:   An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                    training examples
        :param y:   An array of dtype float, shape `(num_examples, num_labels)`, representing the labels of the training
                    examples
        :return:    The classifier or ranker that has been trained
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes a prediction for given test data.

        :param x:   An array of dtype float, shape `(num_examples, num_features)`, representing the features of the test
                    examples
        :return:    An array of dtype float, shape `(num_examples, num_labels)`, representing the labels predicted for
                    the given test examples
        """
        pass
