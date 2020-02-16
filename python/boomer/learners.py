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

    def get_model_name(self) -> str:
        """
        Returns the name that should be used to save the model of the classifier or ranker to a file.

        By default, the model's name is equal to the learner's name as returned by the function `get_name`. This method
        may be overridden if varying names for models should be used.

        :return: The name that should be used to save the model to a file
        """
        return self.get_name()

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns a human-readable name that allows to identify the configuration used by the classifier or ranker.

        :return: The name of the classifier or ranker
        """
        pass

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
