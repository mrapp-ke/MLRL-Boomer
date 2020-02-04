#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides a scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
classification rules. The classifier is composed of several modules, e.g., for rule induction and prediction.
"""
import logging as log
from abc import abstractmethod
from os.path import isdir
from timeit import default_timer as timer

import numpy as np
from boomer.algorithm._head_refinement import HeadRefinement, SingleLabelHeadRefinement, FullHeadRefinement
from boomer.algorithm._losses import Loss, DecomposableLoss, SquaredErrorLoss, LogisticLoss
from boomer.algorithm._pruning import Pruning, IREP
from sklearn.utils.validation import check_is_fitted

from boomer.algorithm._shrinkage import Shrinkage, ConstantShrinkage
from boomer.algorithm._sub_sampling import FeatureSubSampling, RandomFeatureSubsetSelection
from boomer.algorithm._sub_sampling import InstanceSubSampling, Bagging, RandomInstanceSubsetSelection
from boomer.algorithm._sub_sampling import LabelSubSampling, RandomLabelSubsetSelection
from boomer.algorithm.model import Theory, DTYPE_FLOAT32
from boomer.algorithm.persistence import ModelPersistence
from boomer.algorithm.prediction import Prediction, Sign, LinearCombination
from boomer.algorithm.rule_induction import RuleInduction, GradientBoosting
from boomer.algorithm.stats import Stats
from boomer.algorithm.stopping_criteria import SizeStoppingCriterion, TimeStoppingCriterion
from boomer.learners import MLLearner


class MLRuleLearner(MLLearner):
    """
    A scikit-multilearn implementation of a rule learner algorithm for multi-label classification or ranking.

    Attributes
        stats_          Statistics about the training data set
        theory_         The theory that contains the classification rules
        persistence     The 'ModelPersistence' to be used to load/save the theory
    """

    STEP_INITIALIZATION = 0

    STEP_RULE_INDUCTION = 1

    PREFIX_RULES = 'rules'

    stats_: Stats

    theory_: Theory

    def __init__(self, model_dir: str):
        """
        :param model_dir: The path of the directory where models should be stored / loaded from
        """
        super().__init__()
        self.model_dir = model_dir
        self.require_dense = [True, True]  # We need a dense representation of the training data

    @abstractmethod
    def _create_prediction(self) -> Prediction:
        """
        Must be implemented by subclasses in order to create the `Prediction` to be used for making predictions.

        :return: The `Prediction` that has been created
        """
        pass

    @abstractmethod
    def _create_rule_induction(self, stats: Stats) -> RuleInduction:
        """
        Must be implemented by subclasses in order to create the `RuleInduction` to be used for inducing rules.

        :param stats:   Statistics about the training data set
        :return:        The `RuleInduction` that has been created
        """
        pass

    def __create_persistence(self) -> ModelPersistence:
        """
        Creates and returns the [ModelPersistence] that is used to store / load models.

        :return: The [ModelPersistence] that has been created
        """
        model_dir = str(self.model_dir)

        if model_dir is None:
            return None
        elif isdir(model_dir):
            return ModelPersistence(model_dir=model_dir)
        raise ValueError('Invalid value given for parameter \'model_dir\': ' + str(model_dir))

    def __load_rules(self, persistence: ModelPersistence):
        """
        Loads the theory from disk, if available.

        :param persistence: The [ModelPersistence] that should be used
        :return: The loaded theory, as well as the next step to proceed with
        """
        step = MLRuleLearner.STEP_RULE_INDUCTION

        if persistence is not None:
            theory = persistence.load_model(model_name=self.get_model_name(),
                                            file_name_suffix=MLRuleLearner.PREFIX_RULES, fold=self.fold)
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
        # Create a dense representation of the training data
        x = self._ensure_input_format(x)
        y = self._ensure_input_format(y)

        # Obtain information about the training data
        stats = Stats.create_stats(x, y)
        self.stats_ = stats

        # Load theory from disk, if possible
        persistence = self.__create_persistence()
        model, step = self.__load_rules(persistence)

        if model is not None:
            theory = model

        if step == MLRuleLearner.STEP_INITIALIZATION:
            log.info('Inducing classification rules...')
            start_time = timer()

            # Induce rules
            rule_induction = self._create_rule_induction(stats)
            rule_induction.random_state = self.random_state
            theory = rule_induction.induce_rules(stats, x, y, theory)

            # Save theory to disk
            self.__save_rules(persistence, theory)

            end_time = timer()
            run_time = end_time - start_time
            num_candidates = len(theory)
            log.info('%s classification rules induced in %s seconds', num_candidates, run_time)

        return theory

    def __save_rules(self, persistence: ModelPersistence, theory: Theory):
        """
        Saves a theory to disk.

        :param persistence: The [ModelPersistence] that should be used
        :param theory:      The theory to be saved
        """

        if persistence is not None:
            persistence.save_model(theory, model_name=self.get_model_name(),
                                   file_name_suffix=MLRuleLearner.PREFIX_RULES, fold=self.fold)

    def fit(self, x: np.ndarray, y: np.ndarray) -> MLLearner:
        self.theory_ = self._induce_rules(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self)

        # Create a dense representation of the given examples
        x = self._ensure_input_format(x)

        # Convert feature matrix into Fortran-contiguous array
        x = np.asfortranarray(x, dtype=DTYPE_FLOAT32)

        log.info("Making a prediction for %s query instances...", np.shape(x)[0])
        prediction = self._create_prediction()
        prediction.random_state = self.random_state
        return prediction.predict(self.stats_, self.theory_, x)

    def get_params(self, deep=True):
        return {
            'model_dir': self.model_dir
        }

    def set_params(self, **parameters):
        params = self.get_params()
        for parameter, value in parameters.items():
            if parameter in params.keys():
                setattr(self, parameter, value)
            else:
                raise ValueError('Invalid parameter: ' + str(parameter))
        return self

    @abstractmethod
    def get_name(self) -> str:
        pass


class Boomer(MLRuleLearner):
    """
    A scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
    classification rules.
    """

    def __init__(self, model_dir: str = None, num_rules: int = 500, time_limit: int = -1, head_refinement: str = None,
                 loss: str = 'squared-error-loss', label_sub_sampling: int = -1, instance_sub_sampling: str = None,
                 feature_sub_sampling: str = None, pruning: str = None, shrinkage: float = 1.0,
                 l2_regularization_weight: float = 0.0):
        """
        :param num_rules:                   The number of rules to be induced (including the default rule)
        :param time_limit:                  The duration in seconds after which the induction of rules should be
                                            canceled
        :param head_refinement:             The strategy that is used to find the heads of rules. Must be
                                            `single-label`, `full` or None, if the default strategy should be used
        :param loss:                        The loss function to be minimized. Must be `squared-error-loss` or
                                            `logistic-loss`
        :param label_sub_sampling:          The number of samples to be used for sub-sampling the labels each time a new
                                            classification rule is learned. Must be at least 1 or -1, if no sub-sampling
                                            should be used
        :param instance_sub_sampling:       The strategy that is used for sub-sampling the training examples each time a
                                            new classification rule is learned. Must be `bagging`,
                                            `random-instance-selection` or None, if no sub-sampling should be used
        :param feature_sub_sampling:        The strategy that is used for sub-sampling the features each time a
                                            classification rule is refined. Must be `random-feature-selection` or None,
                                            if no sub-sampling should be used
        :param pruning:                     The strategy that is used for pruning rules. Must be `irep` or None, if no
                                            pruning should be used
        :param shrinkage:                   The shrinkage parameter that should be applied to the predictions of newly
                                            induced rules to reduce their effect on the entire model. Must be in (0, 1]
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            scores that are predicted by rules. Must be at least 0
        """
        super().__init__(model_dir)
        self.num_rules = num_rules
        self.time_limit = time_limit
        self.head_refinement = head_refinement
        self.loss = loss
        self.label_sub_sampling = label_sub_sampling
        self.instance_sub_sampling = instance_sub_sampling
        self.feature_sub_sampling = feature_sub_sampling
        self.pruning = pruning
        self.shrinkage = shrinkage
        self.l2_regularization_weight = l2_regularization_weight

    def _create_prediction(self) -> Prediction:
        return Sign(LinearCombination())

    def _create_rule_induction(self, stats: Stats) -> RuleInduction:
        num_rules = int(self.num_rules)
        time_limit = int(self.time_limit)
        stopping_criteria = []

        if num_rules != -1:
            if num_rules > 0:
                stopping_criteria.append(SizeStoppingCriterion(num_rules))
            else:
                raise ValueError('Invalid value given for parameter \'num_rules\': ' + str(num_rules))

        if time_limit != -1:
            if time_limit > 0:
                stopping_criteria.append(TimeStoppingCriterion(time_limit))
            else:
                raise ValueError('Invalid value given for parameter \'time_limit\': ' + str(time_limit))

        l2_regularization_weight = float(self.l2_regularization_weight)

        if l2_regularization_weight < 0:
            raise ValueError(
                'Invalid value given for parameter \'l2_regularization_weight\': ' + str(l2_regularization_weight))

        loss = self.__create_loss(l2_regularization_weight)
        head_refinement = self.__create_head_refinement(loss)
        label_sub_sampling = self.__create_label_sub_sampling(stats)
        instance_sub_sampling = self.__create_instance_sub_sampling()
        feature_sub_sampling = self.__create_feature_sub_sampling()
        pruning = self.__create_pruning()
        shrinkage = self.__create_shrinkage()
        return GradientBoosting(head_refinement, loss, label_sub_sampling, instance_sub_sampling, feature_sub_sampling,
                                pruning, shrinkage, *stopping_criteria)

    def __create_loss(self, l2_regularization_weight: float) -> Loss:
        loss = self.loss

        if loss == 'squared-error-loss':
            return SquaredErrorLoss(l2_regularization_weight)
        elif loss == 'logistic-loss':
            return LogisticLoss(l2_regularization_weight)
        raise ValueError('Invalid value given for parameter \'loss\': ' + str(loss))

    def __create_head_refinement(self, loss: Loss) -> HeadRefinement:
        head_refinement = self.head_refinement

        if head_refinement is None:
            return SingleLabelHeadRefinement() if isinstance(loss, DecomposableLoss) else FullHeadRefinement()
        elif head_refinement == 'single-label':
            return SingleLabelHeadRefinement()
        elif head_refinement == 'full':
            return FullHeadRefinement()
        raise ValueError('Invalid value given for parameter \'head_refinement\': ' + str(head_refinement))

    def __create_label_sub_sampling(self, stats: Stats) -> LabelSubSampling:
        label_sub_sampling = int(self.label_sub_sampling)

        if label_sub_sampling == -1:
            return None
        elif label_sub_sampling > 0:
            if label_sub_sampling < stats.num_labels:
                return RandomLabelSubsetSelection(label_sub_sampling)
            else:
                raise ValueError('Value given for parameter \'label_sub_sampling\' (' + str(label_sub_sampling)
                                 + ') exceeds number of labels in the training data set (' + str(stats.num_labels)
                                 + ')')
        raise ValueError('Invalid value given for parameter \'label_sub_sampling\': ' + str(label_sub_sampling))

    def __create_instance_sub_sampling(self) -> InstanceSubSampling:
        instance_sub_sampling = self.instance_sub_sampling

        if instance_sub_sampling is None:
            return None
        elif instance_sub_sampling == 'bagging':
            return Bagging()
        elif instance_sub_sampling == 'random-instance-selection':
            return RandomInstanceSubsetSelection()
        raise ValueError('Invalid value given for parameter \'instance_sub_sampling\': ' + str(instance_sub_sampling))

    def __create_feature_sub_sampling(self) -> FeatureSubSampling:
        feature_sub_sampling = self.feature_sub_sampling

        if feature_sub_sampling is None:
            return None
        elif feature_sub_sampling == 'random-feature-selection':
            return RandomFeatureSubsetSelection()
        raise ValueError('Invalid value given for parameter \'feature_sub_sampling\': ' + str(feature_sub_sampling))

    def __create_pruning(self) -> Pruning:
        pruning = self.pruning

        if pruning is None:
            return None
        if pruning == 'irep':
            return IREP()
        raise ValueError('Invalid value given for parameter \'pruning\': ' + str(pruning))

    def __create_shrinkage(self) -> Shrinkage:
        shrinkage = float(self.shrinkage)

        if 0.0 < shrinkage < 1.0:
            return ConstantShrinkage(shrinkage)
        if shrinkage == 1.0:
            return None
        raise ValueError('Invalid value given for parameter \'shrinkage\': ' + str(shrinkage))

    def get_name(self) -> str:
        num_rules = str(self.num_rules)
        head_refinement = str(self.head_refinement)
        loss = str(self.loss)
        label_sub_sampling = str(self.label_sub_sampling)
        instance_sub_sampling = str(self.instance_sub_sampling)
        feature_sub_sampling = str(self.feature_sub_sampling)
        pruning = str(self.pruning)
        shrinkage = str(self.shrinkage)
        return 'num-rules=' + num_rules + '_head-refinement=' + head_refinement + '_loss=' + loss \
               + '_label-sub-sampling=' + label_sub_sampling + '_instance-sub-sampling=' + instance_sub_sampling \
               + '_feature-sub-sampling=' + feature_sub_sampling + '_pruning=' + pruning + '_shrinkage=' + shrinkage

    def get_params(self, deep=True):
        params = super().get_params()
        params.update({
            'num_rules': self.num_rules,
            'time_limit': self.time_limit,
            'head_refinement': self.head_refinement,
            'loss': self.loss,
            'label_sub_sampling': self.label_sub_sampling,
            'instance_sub_sampling': self.instance_sub_sampling,
            'feature_sub_sampling': self.feature_sub_sampling,
            'pruning': self.pruning,
            'shrinkage': self.shrinkage,
            'l2_regularization_weight': self.l2_regularization_weight
        })
        return params
