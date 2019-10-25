#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for making predictions based on rules.
"""
from abc import abstractmethod

import numpy as np
from boomer.algorithm._model import DTYPE_SCORES, DTYPE_FEATURES

from boomer.algorithm.model import Theory
from boomer.algorithm.stats import Stats, get_num_examples
from boomer.learners import Module


class Prediction(Module):
    """
    A module that allows to make predictions using a 'Theory'.
    """

    @abstractmethod
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


class Bipartition(Prediction):
    """
    A base class for all subclasses of the class 'Prediction' that predict binary label vectors.
    """

    @abstractmethod
    def predict(self, stats: Stats, theory: Theory, x: np.ndarray) -> np.ndarray:
        pass


class LinearCombination(Bipartition):
    """
    Predicts the linear combination of rules, i.e., the sum of the scores provided by all covering rules for each label.
    """

    def predict(self, stats: Stats, theory: Theory, x: np.ndarray) -> np.ndarray:
        x = np.asfortranarray(x, dtype=DTYPE_FEATURES)
        prediction = np.asfortranarray(np.zeros((get_num_examples(x), stats.num_labels), dtype=DTYPE_SCORES))

        for rule in theory:
            rule.predict(x, prediction)

        return np.where(prediction > 0, 1, 0)
