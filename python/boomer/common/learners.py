#!/usr/bin/python


"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for implementing single- or multi-label classifiers or rankers.
"""
import logging as log
from abc import ABC, abstractmethod
from os.path import isdir
from timeit import default_timer as timer
from typing import List

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from boomer.common.interfaces import Randomized
from boomer.persistence import ModelPersistence


class NominalAttributeLearner(ABC):
    """
    A base class for all single- or multi-label classifiers or rankers that natively support nominal attributes.
    """

    nominal_attribute_indices: List[int] = None


class Learner(BaseEstimator, Randomized):
    """
    A base class for all single- or multi-label classifiers or rankers.
    """

    def set_params(self, **parameters):
        params = self.get_params()
        for parameter, value in parameters.items():
            if parameter in params.keys():
                setattr(self, parameter, value)
            else:
                raise ValueError('Invalid parameter: ' + str(parameter))
        return self

    def get_model_name(self) -> str:
        """
        Returns the name that should be used to save the model of the classifier or ranker to a file.

        By default, the model's name is equal to the learner's name as returned by the function `get_name`. This method
        may be overridden if varying names for models should be used.

        :return: The name that should be used to save the model to a file
        """
        return self.get_name()

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns a human-readable name that allows to identify the configuration used by the classifier or ranker.

        :return: The name of the classifier or ranker
        """
        pass


class MLLearner(Learner):
    """
    A base class for all multi-label classifiers or rankers.

    Attributes
        fold    The current fold or None, if no cross validation is used
        model_  The model
    """

    fold: int = None

    def __init__(self, model_dir: str):
        """
        :param model_dir: The path of the directory where models should be stored / loaded from
        """
        super().__init__()
        self.model_dir = model_dir

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
            return persistence.load_model(model_name=self.get_model_name(), file_name_suffix=self.get_model_prefix(),
                                          fold=self.fold)

        return None

    def __save_model(self, persistence: ModelPersistence, model):
        """
        Saves a model to disk.

        :param persistence: The [ModelPersistence] that should be used
        :param model:       The model to be saved
        """

        if persistence is not None:
            persistence.save_model(model, model_name=self.get_model_name(), file_name_suffix=self.get_model_prefix(),
                                   fold=self.fold)

    def get_params(self, deep=True):
        return {
            'model_dir': self.model_dir
        }

    def fit(self, x, y):
        # Load theory from disk, if possible
        persistence = self.__create_persistence()
        model = self.__load_model(persistence)

        if model is None:
            log.info('Fitting model...')
            start_time = timer()

            # Fit model
            model = self._fit(x, y)

            # Save model to disk
            self.__save_model(persistence, model)

            end_time = timer()
            run_time = end_time - start_time
            log.info('Successfully fit model in %s seconds', run_time)

        self.model_ = model
        return self

    def predict(self, x):
        check_is_fitted(self)
        log.info("Making a prediction for %s query instances...", x.shape[0])
        return self._predict(x)

    @abstractmethod
    def get_model_prefix(self) -> str:
        """
        Returns the prefix to be used when storing model on disk.

        :return: The prefix
        """
        pass

    @abstractmethod
    def _fit(self, x, y):
        """
        Trains a new model on the given training data.

        :param x:   A numpy.ndarray or scipy.sparse matrix of shape `(num_examples, num_features)`, representing the
                    feature values of the training examples
        :param y:   A numpy.ndarray or scipy.sparse matrix of shape `(num_examples, num_labels)`, representing the
                    labels of the training examples
        :return:    The model that has been trained
        """
        pass

    @abstractmethod
    def _predict(self, x):
        """
        Makes a prediction for given query examples.

        :param x:   A numpy.ndarray or scipy.sparse matrix of shape `(num_examples, num_features)`, representing the
                    feature values of the query examples
        :return:    A numpy.ndarray or scipy.sparse matrix of shape `(num_examples, num_labels)`, representing the
                    labels predicted for the given query examples
        """
        pass
