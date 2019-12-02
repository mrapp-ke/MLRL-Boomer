#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides a scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
classification rules. The classifier is composed of several modules, e.g., for rule induction and prediction.
"""
import logging as log
from copy import copy
from timeit import default_timer as timer

import numpy as np
from boomer.algorithm._head_refinement import HeadRefinement, SingleLabelHeadRefinement, FullHeadRefinement
from boomer.algorithm._losses import Loss, DecomposableLoss, SquaredErrorLoss
from boomer.algorithm._pruning import Pruning
from boomer.algorithm._sub_sampling import InstanceSubSampling, FeatureSubSampling
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from boomer.algorithm.model import Theory
from boomer.algorithm.persistence import ModelPersistence
from boomer.algorithm.prediction import Prediction, Sign, LinearCombination
from boomer.algorithm.rule_induction import RuleInduction, GradientBoosting
from boomer.algorithm.stats import Stats
from boomer.learners import MLLearner, BatchMLLearner


class MLRuleLearner(MLLearner):
    """
    A scikit-multilearn implementation of a rule learner algorithm for multi-label classification or ranking.

    Attributes
        stats           Statistics about the training data set
        theory          The theory that contains the classification rules
        persistence     The 'ModelPersistence' to be used to load/save the theory
    """

    STEP_INITIALIZATION = 0

    STEP_RULE_INDUCTION = 1

    PREFIX_RULES = 'rules'

    stats: Stats

    theory: Theory

    persistence: ModelPersistence = None

    def __init__(self, rule_induction: RuleInduction, prediction: Prediction):
        """
        :param rule_induction:  The module that is used to induce classification rules
        :param prediction:      The module that is used to make a prediction
        """

        super().__init__()
        self.require_dense = [True, True]  # We need a dense representation of the training data
        self.rule_induction = rule_induction
        self.prediction = prediction

    def __validate(self):
        """
        Raises exceptions if the algorithm is not configured properly.
        """

        if self.rule_induction is None:
            raise ValueError('Module \'rule_induction\' may not be None')
        if self.prediction is None:
            raise ValueError('Module \'prediction\' may not be None')

    def __load_rules(self):
        """
        Loads the theory from disk, if available.

        :return: The loaded theory, as well as the next step to proceed with
        """
        step = MLRuleLearner.STEP_RULE_INDUCTION

        if self.persistence is not None:
            theory = self.persistence.load_model(file_name_suffix=MLRuleLearner.PREFIX_RULES, fold=self.fold)
        else:
            theory = None

        if theory is None:
            step = MLRuleLearner.STEP_INITIALIZATION

        return theory, step

    def _induce_rules(self, x: np.ndarray, y: np.ndarray, theory: Theory = None) -> Theory:
        """
        Induces classification rules.

        :param x:       An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                        training examples
        :param y:       An array of dtype float, shape `(num_examples, num_labels)`, representing the labels of the
                        training examples
        :param theory:  An existing theory, the induced classification rules should be added to, or None if a new theory
                        should be created
        :return:        A 'Theory' that contains the induced classification rules
        """
        self.__validate()

        # Create a dense representation of the training data
        x = self._ensure_input_format(x)
        y = self._ensure_input_format(y)

        # Obtain information about the training data
        self.stats = Stats.create_stats(x, y)

        # Load theory from disk, if possible
        model, step = self.__load_rules()

        if model is not None:
            theory = model

        if step == MLRuleLearner.STEP_INITIALIZATION:
            log.info('Inducing classification rules...')
            start_time = timer()

            # Induce rules
            self.rule_induction.random_state = self.random_state
            theory = self.rule_induction.induce_rules(self.stats, x, y, theory)

            # Save theory to disk
            self.__save_rules(theory)

            end_time = timer()
            run_time = end_time - start_time
            num_candidates = len(theory)
            log.info('%s classification rules induced in %s seconds', num_candidates, run_time)

        return theory

    def __save_rules(self, theory: Theory):
        """
        Saves a theory to disk.

        :param theory:  The theory to be saved
        """

        if self.persistence is not None:
            self.persistence.save_model(theory, MLRuleLearner.PREFIX_RULES, fold=self.fold)

    def fit(self, x: np.ndarray, y: np.ndarray) -> MLLearner:
        self.theory = self._induce_rules(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ('theory', 'stats'))

        # Create a dense representation of the given examples
        x = self._ensure_input_format(x)

        log.info("Making a prediction for %s query instances...", np.shape(x)[0])
        self.prediction.random_state = self.random_state
        prediction = self.prediction.predict(self.stats, self.theory, x)
        return prediction


class Boomer(MLRuleLearner, BatchMLLearner):
    """
    A scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
    classification rules.
    """

    def __init__(self, num_rules: int = 100, head_refinement: HeadRefinement = None,
                 loss: Loss = SquaredErrorLoss(), instance_sub_sampling: InstanceSubSampling = None,
                 feature_sub_sampling: FeatureSubSampling = None, pruning: Pruning = None, shrinkage: float = 1):
        """
        :param num_rules:               The number of rules to be induced (including the default rule)
        :param head_refinement:         The strategy that is used to find the heads of rules or None, if the default
                                        strategy should be used
        :param loss:                    The loss function to be minimized
        :param instance_sub_sampling:   The strategy that is used for sub-sampling the training examples each time a new
                                        classification rule is learned
        :param feature_sub_sampling:    The strategy that is used for sub-sampling the features each time a
                                        classification rule is refined
        :param pruning:                 The strategy that is used for pruning rules
        :param shrinkage:               The shrinkage parameter that should be applied to the predictions of newly
                                        induced rules to reduce their effect on the entire model. Must be in (0, 1]
        """
        super().__init__(rule_induction=GradientBoosting(num_rules=num_rules,
                                                         head_refinement=
                                                         (FullHeadRefinement() if isinstance(loss, DecomposableLoss)
                                                          else SingleLabelHeadRefinement()) if head_refinement is None
                                                         else head_refinement,
                                                         loss=loss,
                                                         instance_sub_sampling=instance_sub_sampling,
                                                         feature_sub_sampling=feature_sub_sampling,
                                                         pruning=pruning,
                                                         shrinkage=shrinkage),
                         prediction=Sign(LinearCombination()))

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> BatchMLLearner:
        check_is_fitted(self, ('theory', 'stats'))
        self.theory = self._induce_rules(x, y, theory=self.theory)
        return self

    # noinspection PyUnresolvedReferences
    def predict(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self, 'theory') and len(self.theory) < self.rule_induction.num_rules:
            raise NotFittedError('Not enough rules contained by theory')

        return super().predict(x)

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def copy_classifier(self, **kwargs) -> BatchMLLearner:
        copied_classifier = copy(self)
        copied_classifier.rule_induction = copy(self.rule_induction)
        copied_classifier.rule_induction.num_rules = kwargs.get('num_rules', self.rule_induction.num_rules)
        copied_classifier.persistence = kwargs.get('persistence', self.persistence)

        if hasattr(self, 'theory') and hasattr(self, 'stats'):
            if copied_classifier.rule_induction.num_rules >= self.rule_induction.num_rules:
                copied_classifier.theory = self.theory
                copied_classifier.stats = self.stats

        return copied_classifier
