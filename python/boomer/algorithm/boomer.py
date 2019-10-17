#!/usr/bin/python


"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides a scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
classification rules. The classifier is composed of several modules, e.g., for rule induction and prediction.
"""
import logging as log

import numpy as np
from sklearn.utils.validation import check_is_fitted

from boomer.algorithm.model import Theory
from boomer.algorithm.persistence import ModelPersistence
from boomer.learners import MLLearner


class Module:
    """
    A base class for all modules, the "Boomer" algorithm consists of.

    Attributes
        random_state   The seed to be used by RNGs
    """

    random_state: int = 0


class RuleInduction(Module):
    """
    A module that allows to induce a `Theory`, consisting of several classification rules.
    """

    def induce_rules(self, x: np.ndarray, y: np.ndarray) -> Theory:
        """
        Creates and returns a 'Theory' that contains several candidate rules.

        :param x:       An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                        training examples
        :param y:       An array of dtype float, shape `(num_examples, num_labels)`, representing the labels of the
                        training examples
        :return:        A 'Theory' that contains the generated candidate rules
        """
        pass


class Prediction(Module):
    """
    A module that allows to make predictions using a 'Theory'.
    """

    def predict(self, theory: Theory, x: np.ndarray) -> np.ndarray:
        """Predicts the labels of examples using a specific theory.

        :param theory:  The theory that is used to make predictions
        :param x:       An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                        examples to be classified
        :return:        An array of dtype float, shape `(num_examples, num_labels)', representing the predicted labels
        """
        pass


class Boomer(MLLearner):
    """
    A scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
    classification rules.

    Attributes
        theory          The theory that contains the classification rules
        persistence     The 'ModelPersistence' to be used to load/save the theory
    """

    STEP_INITIALIZATION = 0

    STEP_RULE_INDUCTION = 1

    PREFIX_RULES = 'rules'

    theory: Theory

    persistence: ModelPersistence = None

    def __init__(self, rule_induction: RuleInduction, prediction: Prediction, random_state: int = 0):
        """
        :param rule_induction:  The module that is used to induce classification rules
        :param prediction:      The module that is used to make a prediction
        """

        super().__init__()
        self.require_dense = [True, True]  # We need a dense representation of the training data
        self.rule_induction = rule_induction
        self.prediction = prediction
        self.random_state = random_state

    def __load_model(self):
        """
        Loads the model from disk, if available.

        :return: The loaded model, as well as the next step to proceed with
        """
        step = Boomer.STEP_RULE_INDUCTION
        model = self.__load_rules()

        if model is None:
            step = Boomer.STEP_INITIALIZATION

        return model, step

    def __load_rules(self):
        if self.persistence is not None:
            return self.persistence.load_model(file_name_suffix=Boomer.PREFIX_RULES, fold=self.fold)
        else:
            return None

    def __induce_rules(self, x: np.ndarray, y: np.ndarray) -> Theory:
        """
        Induces classification rules.

        :param x:       An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                        training examples
        :param y:       An array of dtype float, shape `(num_examples, num_labels)`, representing the labels of the
                        training examples
        :return:        A 'Theory' that contains the induced classification rules
        """

        self.rule_induction.random_state = self.random_state
        model = self.rule_induction.induce_rules(x, y)
        self.__save_rules(model)
        return model

    def __save_rules(self, model: Theory):
        if self.persistence is not None:
            self.persistence.save_model(model, Boomer.PREFIX_RULES, fold=self.fold)

    def fit(self, x, y):
        # Create a dense representation of the training data
        x = self._ensure_input_format(x)
        y = self._ensure_input_format(y)

        # Load model from disk, if possible
        model, step = self.__load_model()

        if step == Boomer.STEP_INITIALIZATION:
            log.info('Inducing classification rules...')
            model = self.__induce_rules(x, y)
            num_candidates = len(model)
            log.info('%s classification rules induced in total', num_candidates)

        self.theory = model

    def predict(self, x):
        check_is_fitted(self, ('theory', 'stats'))

        # Create a dense representation of the given examples
        x = self._ensure_input_format(x)

        log.info("Making a prediction for %s query instances...", np.shape(x)[0])
        self.prediction.random_state = self.random_state
        return self.prediction.predict(self.theory, x)
