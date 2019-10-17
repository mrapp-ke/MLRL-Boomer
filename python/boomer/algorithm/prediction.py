#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for making predictions based on rules.
"""
import numpy as np

from boomer.algorithm.model import Theory
from boomer.algorithm.modules import Prediction
from boomer.algorithm.stats import Stats, get_num_examples


class Bipartition(Prediction):
    """
    A base class for all subclasses of the class 'Prediction' that predict binary label vectors.
    """

    def predict(self, stats: Stats, theory: Theory, x: np.ndarray) -> np.ndarray:
        pass


class LinearCombination(Bipartition):
    """
    Predicts the linear combination of rules, i.e., the sum of the scores provided by all covering rules for each label.
    """

    def predict(self, stats: Stats, theory: Theory, x: np.ndarray) -> np.ndarray:
        prediction = np.full((get_num_examples(x), stats.num_labels), 0, dtype=float)

        for rule in theory:
            mask = rule.body.match(x)
            prediction[mask] += rule.head

        return np.where(np.greater(prediction, 0), 1, 0)
