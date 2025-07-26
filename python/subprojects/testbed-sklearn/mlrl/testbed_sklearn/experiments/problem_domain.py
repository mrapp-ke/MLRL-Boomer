"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing different kinds of problem domains to be tacked via the scikit-learn framework.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, override

import numpy as np

from sklearn.base import BaseEstimator

from mlrl.testbed_sklearn.experiments.prediction import Predictor

from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.experiments.problem_domain import ClassificationProblem, ProblemDomain, RegressionProblem


class SkLearnProblem(ProblemDomain, ABC):
    """
    An abstract base class for all classes that represent a specific problem domain to be tackled via the scikit-learn
    framework.
    """

    class PredictorFactory(ABC):
        """
        An abstract base class for all factories that allow to create instances of type `Predictor`.
        """

        @abstractmethod
        def create(self) -> Predictor:
            """
            Creates and returns a new object of type `Predictor`.

            :return: The `Predictor` that has been created
            """

    def __init__(self,
                 base_learner: BaseEstimator,
                 prediction_type: PredictionType,
                 predictor_factory: PredictorFactory,
                 fit_kwargs: Optional[Dict[str, Any]] = None,
                 predict_kwargs: Optional[Dict[str, Any]] = None):
        """
        :param base_learner:        A sklearn estimator to be used in the experiment
        :param prediction_type:     The type of the predictions
        :param predictor_factory:   A `PredictorFactory`
        :param fit_kwargs:          Optional keyword arguments to be passed to the learner when fitting a model
        :param predict_kwargs:      Optional keyword arguments to be passed to the learner when obtaining predictions
                                    from a model
        """
        self.base_learner = base_learner
        self.prediction_type = prediction_type
        self.predictor_factory = predictor_factory
        self.fit_kwargs = fit_kwargs
        self.predict_kwargs = predict_kwargs

    @override
    @property
    def learner_name(self) -> str:
        """
        See :func:`mlrl.testbed.experiments.problem_domain.ProblemDomain.learner_name`
        """
        return type(self.base_learner).__name__

    @property
    @abstractmethod
    def feature_dtype(self) -> Any:
        """
        The data type to be used for features.
        """

    @property
    @abstractmethod
    def output_dtype(self) -> Any:
        """
        The data type to be used for outputs.
        """


class SkLearnClassificationProblem(SkLearnProblem, ClassificationProblem):
    """
    Represents a classification problem to be tackled via the scikit-learn framework.
    """

    @override
    @property
    def feature_dtype(self) -> Any:
        return np.float32

    @override
    @property
    def output_dtype(self) -> Any:
        return np.uint8


class SkLearnRegressionProblem(SkLearnProblem, RegressionProblem):
    """
    Represents a regression problem to be tackled via the scikit-learn framework.
    """

    @override
    @property
    def feature_dtype(self) -> Any:
        return np.float32

    @override
    @property
    def output_dtype(self) -> Any:
        return np.float32
