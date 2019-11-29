#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides a scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
classification rules. The classifier is composed of several modules, e.g., for rule induction and prediction.
"""
import logging as log
from timeit import default_timer as timer

import numpy as np
from boomer.algorithm._head_refinement import HeadRefinement, SingleLabelHeadRefinement, FullHeadRefinement
from boomer.algorithm._losses import Loss, DecomposableLoss, SquaredErrorLoss
from boomer.algorithm._sub_sampling import InstanceSubSampling, FeatureSubSampling
from sklearn.utils.validation import check_is_fitted

from boomer.algorithm.model import Theory
from boomer.algorithm.persistence import ModelPersistence
from boomer.algorithm.prediction import Prediction, Sign, LinearCombination
from boomer.algorithm.rule_induction import RuleInduction, GradientBoosting
from boomer.algorithm.stats import Stats
from boomer.learners import MLLearner


class RuleLearner(MLLearner):
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

    def __load_model(self):
        """
        Loads the model from disk, if available.

        :return: The loaded model, as well as the next step to proceed with
        """
        step = RuleLearner.STEP_RULE_INDUCTION
        model = self.__load_rules()

        if model is None:
            step = RuleLearner.STEP_INITIALIZATION

        return model, step

    def __load_rules(self):
        if self.persistence is not None:
            return self.persistence.load_model(file_name_suffix=RuleLearner.PREFIX_RULES, fold=self.fold)
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
        model = self.rule_induction.induce_rules(self.stats, x, y)
        self.__save_rules(model)
        return model

    def __save_rules(self, model: Theory):
        if self.persistence is not None:
            self.persistence.save_model(model, RuleLearner.PREFIX_RULES, fold=self.fold)

    def fit(self, x, y):
        self.__validate()

        # Create a dense representation of the training data
        x = self._ensure_input_format(x)
        y = self._ensure_input_format(y)

        # Obtain information about the training data
        self.stats = Stats.create_stats(x, y)

        # Load model from disk, if possible
        model, step = self.__load_model()

        if step == RuleLearner.STEP_INITIALIZATION:
            log.info('Inducing classification rules...')
            start_time = timer()
            model = self.__induce_rules(x, y)
            end_time = timer()
            run_time = end_time - start_time
            num_candidates = len(model)
            log.info('%s classification rules induced in %s seconds', num_candidates, run_time)

        self.theory = model
        return self

    def predict(self, x):
        check_is_fitted(self, ('theory', 'stats'))

        # Create a dense representation of the given examples
        x = self._ensure_input_format(x)

        log.info("Making a prediction for %s query instances...", np.shape(x)[0])
        self.prediction.random_state = self.random_state
        prediction = self.prediction.predict(self.stats, self.theory, x)
        return prediction


class Boomer(RuleLearner):
    """
    A scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
    classification rules.
    """

    def __init__(self, num_rules: int = 100, head_refinement: HeadRefinement = None,
                 loss: Loss = SquaredErrorLoss(), instance_sub_sampling: InstanceSubSampling = None,
                 feature_sub_sampling: FeatureSubSampling = None, shrinkage: float = 1):
        """
        :param num_rules:               The number of rules to be induced (including the default rule)
        :param head_refinement:         The strategy that is used to find the heads of rules or None, if the default
                                        strategy should be used
        :param loss:                    The loss function to be minimized
        :param instance_sub_sampling:   The strategy that is used for sub-sampling the training examples each time a new
                                        classification rule is learned
        :param feature_sub_sampling:    The strategy that is used for sub-sampling the features each time a
                                        classification rule is refined
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
                                                         shrinkage=shrinkage),
                         prediction=Sign(LinearCombination()))
