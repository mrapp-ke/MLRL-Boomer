"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for implementing single- or multi-label classifiers or rankers.
"""
from abc import ABC, abstractmethod
from typing import List

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

KWARG_PREDICT_SCORES = 'predict_scores'


class NominalAttributeLearner(ABC):
    """
    A base class for all single- or multi-label classifiers or rankers that natively support nominal attributes.
    """

    nominal_attribute_indices: List[int] = None


class Learner(BaseEstimator, ABC):
    """
    A base class for all single- or multi-label classifiers or rankers.
    """

    def fit(self, x, y, **kwargs):
        """
        Fits a model to given training examples and their corresponding ground truth labels.

        :param x:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores the
                    feature values of the training examples
        :param y:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the
                    labels of the training examples according to the ground truth
        :return:    The fitted learner
        """
        self.model_ = self._fit(x, y, **kwargs)
        return self

    def predict(self, x, **kwargs):
        """
        Obtains and returns predictions for given query examples. If the optional keyword argument `predict_scores` is
        set to `True`, regression scores are obtained instead of binary predictions.

        :param x:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores the
                    feature values of the query examples
        :return:    A `numpy.ndarray` or `scipy.sparse` matrix of shape `(num_examples, num_labels)`, that stores the
                    prediction for individual examples and labels
        """
        check_is_fitted(self)

        if bool(kwargs.get(KWARG_PREDICT_SCORES, False)):
            return self._predict_scores(x, **kwargs)
        else:
            return self._predict_labels(x, **kwargs)

    def predict_proba(self, x, **kwargs):
        """
        Obtains and returns probability estimates for given query examples.

        :param x:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores the
                    feature values of the query examples
        :return:    A `numpy.ndarray` or `scipy.sparse` matrix of shape `(num_examples, num_labels)`, that stores the
                    probabilities for individual examples and labels
        """
        check_is_fitted(self)
        return self._predict_proba(x, **kwargs)

    @abstractmethod
    def _fit(self, x, y, **kwargs):
        """
        Must be implemented by subclasses in order to fit a new model to given training examples and their corresponding
        ground truth labels.

        :param x:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores the
                    feature values of the training examples
        :param y:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the
                    labels of the training examples according to the ground truth
        :return:    The model that has been trained
        """
        pass

    @abstractmethod
    def _predict_labels(self, x, **kwargs):
        """
        Must be implemented by subclasses in order to obtain binary predictions for given query examples.

        :param x:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores the
                    feature values of the query examples
        :return:    A `numpy.ndarray` or `scipy.sparse` matrix of shape `(num_examples, num_labels)`, that stores the
                    prediction for individual examples and labels
        """
        raise RuntimeError('Prediction of binary labels not supported using the current configuration')

    def _predict_scores(self, x, **kwargs):
        """
        May be overridden by subclasses in order to obtain regression scores for given query examples.

        :param x:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores the
                    feature values of the query examples
        :return:    A `numpy.ndarray` or `scipy.sparse` matrix of shape `(num_examples, num_labels)`, that stores the
                    regression scores for individual examples and labels
        """
        raise RuntimeError('Prediction of regression scores not supported using the current configuration')

    def _predict_proba(self, x, **kwargs):
        """
        May be overridden by subclasses in order to obtain probability estimates for given query examples.

        :param x:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores the
                    feature values of the query examples
        :return:    A `numpy.ndarray` or `scipy.sparse` matrix of shape `(num_examples, num_labels)`, that stores the
                    probabilities for individual examples and labels
        """
        raise RuntimeError('Prediction of probabilities not supported using the current configuration')
