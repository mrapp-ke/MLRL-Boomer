#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for making predictions based on rules.
"""
from abc import abstractmethod

import numpy as np
from scipy.sparse import csr_matrix

from boomer.algorithm.model import Theory, DTYPE_UINT8, DTYPE_INTP, DTYPE_FLOAT32, DTYPE_FLOAT64
from boomer.interfaces import Randomized
from boomer.stats import Stats


class Prediction(Randomized):
    """
    A module that allows to make predictions using a 'Theory'.
    """

    @abstractmethod
    def predict(self, stats: Stats, theory: Theory, x: np.ndarray) -> np.ndarray:
        """
        Predicts the labels of examples using a specific theory.

        The feature matrix must be given as a dense np.ndarray.

        :param stats:   Statistics about the training data set
        :param theory:  The theory that is used to make predictions
        :param x:       An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                        examples to be classified
        :return:        An array of dtype float, shape `(num_examples, num_labels)', representing the predicted labels
        """
        pass

    @abstractmethod
    def predict_csr(self, stats: Stats, theory: Theory, x: csr_matrix) -> np.ndarray:
        """
        Predicts the labels of examples using a specific theory.

        :param stats:   Statistics about the training data set
        :param theory:  The theory that is used to make predictions
        :param x:       An csr_matrix of dtype float, shape `(num_examples, num_features)`, representing the features of
                        the examples to be classified
        :return:        An array of dtype float, shape `(num_examples, num_labels)`, representing the predicted labels
        """
        pass


class Ranking(Prediction):
    """
    A base class for all subclasses of the class 'Prediction' that predict numerical scores.
    """

    @abstractmethod
    def predict(self, stats: Stats, theory: Theory, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_csr(self, stats: Stats, theory: Theory, x: csr_matrix) -> np.ndarray:
        pass


class LinearCombination(Ranking):
    """
    Predicts the linear combination of rules, i.e., the sum of the scores provided by all covering rules for each label.
    """

    def predict(self, stats: Stats, theory: Theory, x: np.ndarray) -> np.ndarray:
        predictions = np.zeros((x.shape[0], stats.num_labels), dtype=DTYPE_FLOAT64, order='C')

        for rule in theory:
            rule.predict(x, predictions)

        return predictions

    def predict_csr(self, stats: Stats, theory: Theory, x: csr_matrix) -> np.ndarray:
        predictions = np.zeros((x.shape[0], stats.num_labels), dtype=DTYPE_FLOAT64, order='C')
        x_data = np.ascontiguousarray(x.data, dtype=DTYPE_FLOAT32)
        x_row_indices = np.ascontiguousarray(x.indptr, dtype=DTYPE_INTP)
        x_col_indices = np.ascontiguousarray(x.indices, dtype=DTYPE_INTP)
        num_features = x.shape[1]
        tmp_array1 = np.empty(num_features, dtype=DTYPE_FLOAT32, order='C')
        tmp_array2 = np.zeros(num_features, dtype=DTYPE_INTP, order='C')
        n = 1

        for rule in theory:
            rule.predict_csr(x_data, x_row_indices, x_col_indices, num_features, tmp_array1, tmp_array2, n, predictions)
            n += 1

        return predictions


class Bipartition(Prediction):
    """
    A base class for all subclasses of the class 'Prediction' that predict binary label vectors.
    """

    @abstractmethod
    def predict(self, stats: Stats, theory: Theory, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_csr(self, stats: Stats, theory: Theory, x: csr_matrix) -> np.ndarray:
        pass


class Sign(Bipartition):
    """
    Turns numerical scores into a binary label vector according to the sign function, i.e., 1, if a score is greater
    than zero, 1 otherwise.
    """

    def __init__(self, ranking: Ranking):
        """
        :param ranking: The ranking whose prediction should be turned into a binary label vector
        """
        self.ranking = ranking

    def predict(self, stats: Stats, theory: Theory, x: np.ndarray) -> np.ndarray:
        predictions = self.ranking.predict(stats, theory, x)
        return np.where(predictions > 0, 1, 0)

    def predict_csr(self, stats: Stats, theory: Theory, x: csr_matrix) -> np.ndarray:
        predictions = self.ranking.predict_csr(stats, theory, x)
        return np.where(predictions > 0, 1, 0)


class DecisionList(Prediction):
    """
    Predicts labels according to a list of rules interpreted as a decision list.
    """

    def predict(self, stats: Stats, theory: Theory, x: np.ndarray) -> np.ndarray:
        predictions = np.zeros((x.shape[0], stats.num_labels), dtype=DTYPE_FLOAT64, order='C')
        mask = np.ones((x.shape[0], stats.num_labels), dtype=DTYPE_UINT8, order='C')

        for rule in theory:
            rule.predict(x, predictions, mask)

        return predictions

    def predict_csr(self, stats: Stats, theory: Theory, x: csr_matrix) -> np.ndarray:
        predictions = np.zeros((x.shape[0], stats.num_labels), dtype=DTYPE_FLOAT64, order='C')
        mask = np.ones((x.shape[0], stats.num_labels), dtype=DTYPE_UINT8, order='C')
        x_data = np.ascontiguousarray(x.data, dtype=DTYPE_FLOAT32)
        x_row_indices = np.ascontiguousarray(x.indptr, dtype=DTYPE_INTP)
        x_col_indices = np.ascontiguousarray(x.indices, dtype=DTYPE_INTP)
        num_features = x.shape[1]
        tmp_array1 = np.empty(num_features, dtype=DTYPE_FLOAT32, order='C')
        tmp_array2 = np.zeros(num_features, dtype=DTYPE_INTP, order='C')
        n = 1

        for rule in theory:
            rule.predict_csr(x_data, x_row_indices, x_col_indices, num_features, tmp_array1, tmp_array2, n, predictions,
                             mask)
            n += 1

        return predictions
