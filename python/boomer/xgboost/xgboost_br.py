#!/usr/bin/python
import logging as log
from math import log2, floor

import numpy as np
from sklearn.utils.validation import check_is_fitted
from skmultilearn.problem_transform import BinaryRelevance
from xgboost import XGBClassifier

from boomer.learners import MLLearner


class XGBoostBR(MLLearner):
    br_: BinaryRelevance

    def __init__(self, model_dir: str, learning_rate: float = 1.0, reg_lambda: float = 0.0):
        """
        :param model_dir: The path of the directory where models should be stored / loaded from
        """
        super().__init__()
        self.model_dir = model_dir
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

    def get_params(self, deep=True):
        return {
            'model_dir': self.model_dir,
            'learning_rate': self.learning_rate,
            'reg_lambda': self.reg_lambda
        }

    def get_name(self) -> str:
        return 'xgboost_learning-rate=' + str(self.learning_rate) + '_reg-lambda=' + str(self.reg_lambda)

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'MLLearner':
        learning_rate = float(self.learning_rate)
        reg_lambda = float(self.reg_lambda)
        random_state = self.random_state
        num_features = x.shape[1]
        colsample_bynode = float(floor(log2(num_features - 1) + 1)) / float(num_features)
        xgboost = XGBClassifier(booster='gbtree', learning_rate=learning_rate, n_jobs=1, tree_method='exact',
                                random_state=random_state, subsample=1.0, colsample_bytree=1.0, colsample_bylevel=1.0,
                                colsample_bynode=colsample_bynode, reg_alpha=0.0, reg_lambda=reg_lambda,
                                objective='binary:logistic')
        br = BinaryRelevance(classifier=xgboost)
        br.fit(x, y)
        self.br_ = br
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        log.info("Making a prediction for %s query instances...", np.shape(x)[0])
        br = self.br_
        return br.predict(x)
