"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides mixins that additional functionality to machine learning algorithms.
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from sklearn.base import BaseEstimator as SkLearnBaseEstimator, ClassifierMixin as SkLearnClassifierMixin, \
    MultiOutputMixin as SkLearnMultiOutputMixin, RegressorMixin as SkLearnRegressorMixin
from sklearn.utils.validation import check_is_fitted

KWARG_PREDICT_SCORES = 'predict_scores'


class OrdinalFeatureSupportMixin(ABC):
    """
    A mixin for all machine learning algorithms that natively support ordinal features.
    """

    ordinal_feature_indices: Optional[List[int]] = None

    def set_ordinal_feature_indices(self, indices):
        """
        Sets the indices of all ordinal features.

        :param indices: A `np.ndarray` or `Iterable` that stores the indices to be set
        """
        self.ordinal_feature_indices = None if indices is None else list(indices)


class NominalFeatureSupportMixin(ABC):
    """
    A mixin for all machine learning algorithms that natively support nominal features.
    """

    nominal_feature_indices: Optional[List[int]] = None

    def set_nominal_feature_indices(self, indices):
        """
        Sets the indices of all nominal features.
        
        :param indices: A `np.ndarray` or `Iterable` that stores the indices to be set
        """
        self.nominal_feature_indices = None if indices is None else list(indices)


class IncrementalPredictor(ABC):
    """ 
    A base class for all classes that allow to obtain incremental predictions from a machine learning algorithm.
    """

    def has_next(self) -> bool:
        """
            Returns whether there are any remaining ensemble members that have not been used yet or not.

            :return: True, if there are any remaining ensemble members, False otherwise
            """
        return self.get_num_next() > 0

    @abstractmethod
    def get_num_next(self) -> int:
        """
            Returns the number of remaining ensemble members that have not been used yet.

            :return: The number of remaining ensemble members
            """

    @abstractmethod
    def apply_next(self, step_size: int):
        """
            Updates the current predictions by considering several of the remaining ensemble members. If not enough
            ensemble members are remaining, only the available ones will be used for updating the current predictions.

            :param step_size:   The number of additional ensemble members to be considered for prediction
            :return:            A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray` of shape
                                `(num_examples, num_outputs)`, that stores the updated prediction for individual
                                examples and outputs
            """


class ClassifierMixin(SkLearnBaseEstimator, SkLearnClassifierMixin, SkLearnMultiOutputMixin, ABC):
    """
    A mixin for all machine learning algorithms that can be applied to classification problems.
    """

    # pylint: disable=attribute-defined-outside-init
    def fit(self, x, y, **kwargs):
        """
        Fits a model to given training examples and their corresponding ground truth labels.

        :param x:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_features)`, that stores the feature values of the training examples
        :param y:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_labels)`, that stores the labels of the training examples according to the
                    ground truth
        :return:    The fitted learner
        """
        self.model_ = self._fit(x, y, **kwargs)
        return self

    def predict(self, x, **kwargs):
        """
        Obtains and returns predictions for given query examples. If the optional keyword argument `predict_scores` is
        set to `True`, scores are obtained instead of binary predictions.

        :keyword predict_scores:    True, if scores should be obtained, False, if binary predictions should be obtained
        :param x:                   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                    `(num_examples, num_features)`, that stores the feature values of the query examples
        :return:                    A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray` of shape
                                    `(num_examples, num_labels)`, that stores the prediction for individual examples and
                                    labels
        """
        check_is_fitted(self)

        if bool(kwargs.get(KWARG_PREDICT_SCORES, False)):
            return self._predict_scores(x, **kwargs)
        return self._predict_binary(x, **kwargs)

    def predict_proba(self, x, **kwargs):
        """
        Obtains and returns probability estimates for given query examples.

        :param x:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_features)`, that stores the feature values of the query examples
        :return:    A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray` of shape
                    `(num_examples, num_labels)`, that stores the probabilities for individual examples and labels
        """
        check_is_fitted(self)
        return self._predict_proba(x, **kwargs)

    @abstractmethod
    def _fit(self, x, y, **kwargs):
        """
        Must be implemented by subclasses in order to fit a new model to given training examples and their corresponding
        ground truth labels.

        :param x:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_features)`, that stores the feature values of the training examples
        :param y:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_labels)`, that stores the labels of the training examples according to the
                    ground truth
        :return:    The model that has been trained
        """

    def _predict_scores(self, x, **kwargs):
        """
        May be overridden by subclasses in order to obtain scores for given query examples.

        :param x:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_features)`, that stores the feature values of the query examples
        :return:    A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_labels)`, that stores the scores for individual examples and labels
        """
        raise RuntimeError('Prediction of scores not supported using the current configuration')

    def _predict_proba(self, x, **kwargs):
        """
        May be overridden by subclasses in order to obtain probability estimates for given query examples.

        :param x:   A `numpy.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores the
                    feature values of the query examples
        :return:    A `numpy.ndarray` or `scipy.sparse` matrix of shape `(num_examples, num_labels)`, that stores the
                    probabilities for individual examples and labels
        """
        raise RuntimeError('Prediction of probabilities not supported using the current configuration')

    def _predict_binary(self, x, **kwargs):
        """
        May be overridden by subclasses in order to obtain binary predictions for given query examples.

        :param x:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_features)`, that stores the feature values of the query examples
        :return:    A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray` of shape
                    `(num_examples, num_labels)`, that stores the prediction for individual examples and labels
        """
        raise RuntimeError('Prediction of binary labels not supported using the current configuration')


class IncrementalClassifierMixin(ABC):
    """
    A mixin for all machine learning algorithms that can be applied to classification problems and support incremental
    prediction. For example, when dealing with ensemble models that consist of several ensemble members, it is possible
    to consider only a subset of the ensemble members for prediction.
    """

    def predict_incrementally(self, x, **kwargs) -> IncrementalPredictor:
        """
        Returns an `IncrementalPredictor` that allows to obtain predictions for given query examples incrementally.

        :param x:                   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                    `(num_examples, num_features)`, that stores the feature values of the query examples
        :keyword predict_scores:    True, if scores should be obtained, False, if binary predictions should be obtained
        :return:                    The `IncrementalPredictor` that has been created
        """
        check_is_fitted(self)

        if bool(kwargs.get(KWARG_PREDICT_SCORES, False)):
            return self._predict_scores_incrementally(x, **kwargs)
        return self._predict_binary_incrementally(x, **kwargs)

    def predict_proba_incrementally(self, x, **kwargs) -> IncrementalPredictor:
        """
        Returns an `IncrementalPredictor` that allows to obtain probability estimates for given query examples
        incrementally.

        :param x:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_features)`, that stores the feature values of the query examples
        :return:    The `IncrementalPredictor` that has been created
        """
        check_is_fitted(self)
        return self._predict_proba_incrementally(x, **kwargs)

    def _predict_binary_incrementally(self, x, **kwargs) -> IncrementalPredictor:
        """
        May be overridden by subclasses in order to create an `IncrementalPredictor` that allows to obtain binary
        predictions for given query examples incrementally.

        :param x:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_features)`, that stores the feature values of the query examples
        :return:    The `IncrementalPredictor` that has been created
        """
        raise RuntimeError('Incremental prediction of binary labels not supported using the current configuration')

    def _predict_scores_incrementally(self, x, **kwargs) -> IncrementalPredictor:
        """
        May be overridden by subclasses in order to create an `IncrementalPredictor` that allows to obtain scores for
        given query examples incrementally.

        :param x:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_features)`, that stores the feature values of the query examples
        :return:    The `IncrementalPredictor` that has been created
        """
        raise RuntimeError('Incremental prediction of scores not supported using the current configuration')

    def _predict_proba_incrementally(self, x, **kwargs) -> IncrementalPredictor:
        """
        May be overridden by subclasses in order to create an `IncrementalPredictor` that allows to obtain probability
        estimates for given query examples incrementally.

        :param x:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_features)`, that stores the feature values of the query examples
        :return:    The `IncrementalPredictor` that has been created
        """
        raise RuntimeError('Incremental prediction of probabilities not supported using the current configuration')


class RegressorMixin(SkLearnBaseEstimator, SkLearnRegressorMixin, SkLearnMultiOutputMixin, ABC):
    """
    A mixin for all machine learning algorithms that can be applied to regression problems.
    """

    # pylint: disable=attribute-defined-outside-init
    def fit(self, x, y, **kwargs):
        """
        Fits a model to given training examples and their corresponding ground truth regression scores.

        :param x:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_features)`, that stores the feature values of the training examples
        :param y:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_outputs)`, that stores the regression scores of the training examples according
                    to the ground truth
        :return:    The fitted learner
        """
        self.model_ = self._fit(x, y, **kwargs)
        return self

    def predict(self, x, **kwargs):
        """
        Obtains and returns predictions for given query examples.

        :param x:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_features)`, that stores the feature values of the query examples
        :return:    A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray` of shape
                    `(num_examples, num_outputs)`, that stores the prediction for individual examples and outputs
        """
        check_is_fitted(self)
        return self._predict_scores(x, **kwargs)

    @abstractmethod
    def _fit(self, x, y, **kwargs):
        """
        Must be implemented by subclasses in order to fit a new model to given training examples and their corresponding
        ground truth regression scores.

        :param x:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_features)`, that stores the feature values of the training examples
        :param y:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_outputs)`, that stores the regression scores of the training examples according
                    to the ground truth
        :return:    The model that has been trained
        """

    def _predict_scores(self, x, **kwargs):
        """
        May be overridden by subclasses in order to obtain scores for given query examples.

        :param x:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_features)`, that stores the feature values of the query examples
        :return:    A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_outputs)`, that stores the scores for individual examples and outputs
        """
        raise RuntimeError('Prediction of scores not supported using the current configuration')


class IncrementalRegressorMixin(ABC):
    """
    A mixin for all machine learning algorithms that can be applied to regression problems and support incremental
    prediction. For example, when dealing with ensemble models that consist of several ensemble members, it is possible
    to consider only a subset of the ensemble members for prediction.
    """

    def predict_incrementally(self, x, **kwargs) -> IncrementalPredictor:
        """
        Returns an `IncrementalPredictor` that allows to obtain predictions for given query examples incrementally.

        :param x:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_features)`, that stores the feature values of the query examples
        :return:    The `IncrementalPredictor` that has been created
        """
        check_is_fitted(self)
        return self._predict_scores_incrementally(x, **kwargs)

    def _predict_scores_incrementally(self, x, **kwargs) -> IncrementalPredictor:
        """
        May be overridden by subclasses in order to create an `IncrementalPredictor` that allows to obtain scores for
        given query examples incrementally.

        :param x:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_features)`, that stores the feature values of the query examples
        :return:    The `IncrementalPredictor` that has been created
        """
        raise RuntimeError('Incremental prediction of scores not supported using the current configuration')
