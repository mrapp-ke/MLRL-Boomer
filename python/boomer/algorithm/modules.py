#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for the modules the BOOMER algorithm consists of.
"""
import numpy as np

from boomer.algorithm.model import Theory
from boomer.algorithm.stats import Stats


class Module:
    """
    A base class for all modules, the "Boomer" algorithm consists of.

    Attributes
        random_state   The seed to be used by RNGs
    """

    random_state: int = 0


class RuleInduction(Module):
    """
    A module that allows to induce a `Theory`, consisting of several classification rules.
    """

    def induce_rules(self, stats: Stats, x: np.ndarray, y: np.ndarray) -> Theory:
        """
        Creates and returns a 'Theory' that contains several candidate rules.

        :param stats:   Statistics about the training data set
        :param x:       An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                        training examples
        :param y:       An array of dtype float, shape `(num_examples, num_labels)`, representing the labels of the
                        training examples
        :return:        A 'Theory' that contains the generated candidate rules
        """
        pass


class Prediction(Module):
    """
    A module that allows to make predictions using a 'Theory'.
    """

    def predict(self, stats: Stats, theory: Theory, x: np.ndarray) -> np.ndarray:
        """
        Predicts the labels of examples using a specific theory.

        :param stats:   Statistics about the training data set
        :param theory:  The theory that is used to make predictions
        :param x:       An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                        examples to be classified
        :return:        An array of dtype float, shape `(num_examples, num_labels)', representing the predicted labels
        """
        pass
