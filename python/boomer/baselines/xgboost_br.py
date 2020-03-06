#!/usr/bin/python
import logging as log
from math import log2, floor
from os.path import isdir
from timeit import default_timer as timer

import numpy as np
from sklearn.utils.validation import check_is_fitted
from skmultilearn.problem_transform import BinaryRelevance
from xgboost import XGBClassifier

from boomer.algorithm.persistence import ModelPersistence
from boomer.learners import MLLearner


class XGBoostBR(MLLearner):
    PREFIX_BINARY_RELEVANCE: str = 'br'

    br_: BinaryRelevance

    def __init__(self, model_dir: str, learning_rate: float = 1.0, reg_lambda: float = 0.0):
        """
        :param model_dir: The path of the directory where models should be stored / loaded from
        """
        super().__init__()
        self.model_dir = model_dir
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

    def __create_persistence(self) -> ModelPersistence:
        """
        Creates and returns the [ModelPersistence] that is used to store / load models.

        :return: The [ModelPersistence] that has been created
        """
        model_dir = self.model_dir

        if model_dir is None:
            return None
        elif isdir(model_dir):
            return ModelPersistence(model_dir=model_dir)
        raise ValueError('Invalid value given for parameter \'model_dir\': ' + str(model_dir))

    def __load_model(self, persistence: ModelPersistence):
        """
        Loads the model from disk, if available.

        :param persistence: The [ModelPersistence] that should be used
        :return:            The loaded model
        """
        if persistence is not None:
            return persistence.load_model(model_name=self.get_model_name(),
                                          file_name_suffix=XGBoostBR.PREFIX_BINARY_RELEVANCE, fold=self.fold)

        return None

    def __save_model(self, persistence: ModelPersistence, model: BinaryRelevance):
        """
        Saves a model to disk.

        :param persistence: The [ModelPersistence] that should be used
        :param model:       The model to be saved
        """

        if persistence is not None:
            persistence.save_model(model, model_name=self.get_model_name(),
                                   file_name_suffix=XGBoostBR.PREFIX_BINARY_RELEVANCE, fold=self.fold)

    def get_params(self, deep=True):
        return {
            'model_dir': self.model_dir,
            'learning_rate': self.learning_rate,
            'reg_lambda': self.reg_lambda
        }

    def get_name(self) -> str:
        return 'xgboost_learning-rate=' + str(self.learning_rate) + '_reg-lambda=' + str(self.reg_lambda)

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'MLLearner':
        # Load model from disk, if possible
        persistence = self.__create_persistence()
        model = self.__load_model(persistence)

        if model is None:
            log.info('Training XGBoost model...')
            start_time = timer()

            learning_rate = float(self.learning_rate)
            reg_lambda = float(self.reg_lambda)
            random_state = self.random_state
            num_features = x.shape[1]
            colsample_bynode = float(floor(log2(num_features - 1) + 1)) / float(num_features)
            xgboost = XGBClassifier(booster='gbtree', learning_rate=learning_rate, n_jobs=1, tree_method='exact',
                                    random_state=random_state, subsample=1.0, colsample_bytree=1.0,
                                    colsample_bylevel=1.0, colsample_bynode=colsample_bynode, reg_alpha=0.0,
                                    reg_lambda=reg_lambda, objective='binary:logistic')
            model = BinaryRelevance(classifier=xgboost)
            model.fit(x, y)

            # Save model to disk
            self.__save_model(persistence, model)

            end_time = timer()
            run_time = end_time - start_time
            log.info('XGBoost model trained in %s seconds', run_time)

        self.br_ = model
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        log.info("Making a prediction for %s query instances...", np.shape(x)[0])
        br = self.br_
        return br.predict(x)
