#!/usr/bin/python
from abc import abstractmethod
from math import log2, floor

import numpy as np
from skmultilearn.problem_transform import BinaryRelevance
from xgboost import XGBClassifier

from boomer.learners import MLLearner
from boomer.stats import Stats


class BRLearner(MLLearner):
    """
    A base class for all learners that use the binary relevance transformation method.
    """

    def __init__(self, model_dir: str):
        super().__init__(model_dir)

    def _get_model_prefix(self) -> str:
        return 'br'

    def _fit(self, stats: Stats, x: np.ndarray, y: np.ndarray, random_state: int):
        base_learner = self._create_base_learner(stats, random_state)
        br = BinaryRelevance(classifier=base_learner)
        br.fit(x, y)
        return br

    def _predict(self, model, stats: Stats, x: np.ndarray, random_state: int) -> np.ndarray:
        return model.predict(x)

    @abstractmethod
    def _create_base_learner(self, stats: Stats, random_state: int):
        """
        Must be implemented by subclasses to create the base learner that should be used.

        :param stats:           Statistics about the training data set
        :param random_state:    The seed to be used by RNGs
        :return:                The base learner that has been created
        """
        pass


class XGBoost(BRLearner):
    """
    XGBOOST using the binary relevance transformation method.
    """

    def __init__(self, model_dir: str, learning_rate: float = 1.0, reg_lambda: float = 0.0):
        """
        :param learning_rate:   The learning rate
        :param reg_lambda:      The L2 regularization weight
        """
        super().__init__(model_dir)
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

    def get_params(self, deep=True):
        params = super().get_params()
        params.update({
            'learning_rate': self.learning_rate,
            'reg_lambda': self.reg_lambda
        })
        return params

    def get_name(self) -> str:
        return 'xgboost_learning-rate=' + str(self.learning_rate) + '_reg-lambda=' + str(self.reg_lambda)

    def _create_base_learner(self, stats: Stats, random_state: int):
        learning_rate = float(self.learning_rate)
        reg_lambda = float(self.reg_lambda)
        num_features = stats.num_features
        colsample_bynode = float(floor(log2(num_features - 1) + 1)) / float(num_features)
        return XGBClassifier(booster='gbtree', learning_rate=learning_rate, n_jobs=1, tree_method='exact',
                             random_state=random_state, subsample=1.0, colsample_bytree=1.0,
                             colsample_bylevel=1.0, colsample_bynode=colsample_bynode, reg_alpha=0.0,
                             reg_lambda=reg_lambda, objective='binary:logistic')
