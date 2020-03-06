#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Implements different problem transformation methods.
"""
from abc import abstractmethod

import numpy as np
from skmultilearn.base.problem_transformation import ProblemTransformationBase
from skmultilearn.problem_transform import BinaryRelevance

from boomer.learners import MLLearner, Learner
from boomer.stats import Stats


class ProblemTransformationLearner(MLLearner):
    """
    A base class for all multi-label classifiers or rankers that use a problem transformation method.
    """

    def __init__(self, model_dir: str, base_learner: Learner):
        """
        :param base_learner: The base learner to be used by the problem transformation method
        """
        super().__init__(model_dir)
        self.base_learner = base_learner

    def get_params(self, deep=True):
        params = super().get_params()
        params.update({
            'base_learner': self.base_learner
        })
        return params

    def _fit(self, stats: Stats, x: np.ndarray, y: np.ndarray, random_state: int):
        base_learner = self.base_learner
        transformation_method = self._create_transformation_method(base_learner)
        transformation_method.classifier = base_learner
        transformation_method.fit(x, y)
        return transformation_method

    def _predict(self, model, stats: Stats, x: np.ndarray, random_state: int) -> np.ndarray:
        return model.predict(x)

    def get_name(self) -> str:
        return self.base_learner.get_name()

    @abstractmethod
    def _create_transformation_method(self, base_learner: Learner) -> ProblemTransformationBase:
        pass

    @abstractmethod
    def _get_model_prefix(self) -> str:
        pass


class BRLearner(ProblemTransformationLearner):
    """
    A multi-label classifier or ranker that uses the binary relevance method.
    """

    def __init__(self, model_dir: str, base_learner: Learner):
        super().__init__(model_dir, base_learner)

    def _create_transformation_method(self, base_learner: Learner) -> ProblemTransformationBase:
        return BinaryRelevance(classifier=base_learner)

    def _get_model_prefix(self) -> str:
        return 'br'
