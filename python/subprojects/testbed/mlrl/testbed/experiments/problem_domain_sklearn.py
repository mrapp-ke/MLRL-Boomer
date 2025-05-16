"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing different kinds of problem domains.
"""
from typing import Any, Callable, Dict, Optional

from sklearn.base import BaseEstimator

from mlrl.testbed.experiments.input.dataset.splitters.splitter import DatasetSplitter
from mlrl.testbed.experiments.prediction import Predictor
from mlrl.testbed.experiments.problem_domain import ProblemDomain
from mlrl.testbed.experiments.problem_type import ProblemType


class SkLearnProblem(ProblemDomain):
    """
    An abstract base class for all classes that implement a specific problem domain to be tackled via the scikit-learn
    framework.
    """

    PredictorFactory = Callable[[], Predictor]

    def __init__(self,
                 problem_type: ProblemType,
                 dataset_splitter: DatasetSplitter,
                 base_learner: BaseEstimator,
                 predictor_factory: PredictorFactory,
                 fit_kwargs: Optional[Dict[str, Any]] = None,
                 predict_kwargs: Optional[Dict[str, Any]] = None):
        """
        :param problem_type:        The type of the machine learning problem
        :param dataset_splitter:    The method to be used for splitting the dataset into training and test datasets
        :param base_learner:        A sklearn estimator to be used in the experiment
        :param predictor_factory:   A `PredictorFactory`
        :param fit_kwargs:          Optional keyword arguments to be passed to the learner when fitting a model
        :param predict_kwargs:      Optional keyword arguments to be passed to the learner when obtaining predictions
                                    from a model
        """
        super().__init__(problem_type=problem_type, dataset_splitter=dataset_splitter)
        self.base_learner = base_learner
        self.predictor_factory = predictor_factory
        self.fit_kwargs = fit_kwargs
        self.predict_kwargs = predict_kwargs

    @property
    def learner_name(self) -> str:
        """
        See :func:`mlrl.testbed.experiments.problem_domain.ProblemDomain.learner_name`
        """
        return type(self.base_learner).__name__
