#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides a scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
classification rules. The classifier is composed of several modules, e.g., for rule induction and prediction.
"""
from abc import abstractmethod
from typing import List

import numpy as np
from boomer.algorithm._example_wise_losses import ExampleWiseLogisticLoss
from boomer.algorithm._head_refinement import HeadRefinement, SingleLabelHeadRefinement, FullHeadRefinement
from boomer.algorithm._heuristics import Heuristic, HammingLoss, Precision
from boomer.algorithm._label_wise_losses import LabelWiseSquaredErrorLoss, LabelWiseLogisticLoss
from boomer.algorithm._label_wise_measure import LabelWiseMeasure
from boomer.algorithm._losses import Loss, DecomposableLoss
from boomer.algorithm._pruning import Pruning, IREP
from boomer.algorithm._shrinkage import Shrinkage, ConstantShrinkage
from boomer.algorithm._sub_sampling import FeatureSubSampling, RandomFeatureSubsetSelection
from boomer.algorithm._sub_sampling import InstanceSubSampling, Bagging, RandomInstanceSubsetSelection
from boomer.algorithm._sub_sampling import LabelSubSampling, RandomLabelSubsetSelection

from boomer.algorithm.model import DTYPE_FLOAT32
from boomer.algorithm.prediction import Prediction, Sign, LinearCombination, DecisionList
from boomer.algorithm.rule_induction import RuleInduction, GradientBoosting, SeparateAndConquer
from boomer.algorithm.stopping_criteria import StoppingCriterion, SizeStoppingCriterion, TimeStoppingCriterion, \
    UncoveredLabelsCriterion
from boomer.learners import MLLearner
from boomer.stats import Stats

HEAD_REFINEMENT_SINGLE = 'single-label'

HEAD_REFINEMENT_FULL = 'full'

LOSS_LABEL_WISE_LOGISTIC = 'label-wise-logistic-loss'

LOSS_LABEL_WISE_SQUARED_ERROR = 'label-wise-squared-error-loss'

LOSS_EXAMPLE_WISE_LOGISTIC = 'example-wise-logistic-loss'

MEASURE_LABEL_WISE = 'label-wise-measure'

HEURISTIC_PRECISION = 'precision'

HEURISTIC_HAMMING_LOSS = 'hamming-loss'

INSTANCE_SUB_SAMPLING_RANDOM = 'random-instance-selection'

INSTANCE_SUB_SAMPLING_BAGGING = 'bagging'

FEATURE_SUB_SAMPLING_RANDOM = 'random-feature-selection'

PRUNING_IREP = 'irep'


def _create_label_sub_sampling(label_sub_sampling: int, stats: Stats) -> LabelSubSampling:
    if label_sub_sampling == -1:
        return None
    elif label_sub_sampling > 0:
        if label_sub_sampling < stats.num_labels:
            return RandomLabelSubsetSelection(label_sub_sampling)
        else:
            raise ValueError('Value given for parameter \'label_sub_sampling\' (' + str(label_sub_sampling)
                             + ') must be less that the number of labels in the training data set ('
                             + str(stats.num_labels) + ')')
    raise ValueError('Invalid value given for parameter \'label_sub_sampling\': ' + str(label_sub_sampling))


def _create_instance_sub_sampling(instance_sub_sampling: str) -> InstanceSubSampling:
    if instance_sub_sampling is None:
        return None
    elif instance_sub_sampling == INSTANCE_SUB_SAMPLING_BAGGING:
        return Bagging()
    elif instance_sub_sampling == INSTANCE_SUB_SAMPLING_RANDOM:
        return RandomInstanceSubsetSelection()
    raise ValueError('Invalid value given for parameter \'instance_sub_sampling\': ' + str(instance_sub_sampling))


def _create_feature_sub_sampling(feature_sub_sampling: str) -> FeatureSubSampling:
    if feature_sub_sampling is None:
        return None
    elif feature_sub_sampling == FEATURE_SUB_SAMPLING_RANDOM:
        return RandomFeatureSubsetSelection()
    raise ValueError('Invalid value given for parameter \'feature_sub_sampling\': ' + str(feature_sub_sampling))


def _create_pruning(pruning: str) -> Pruning:
    if pruning is None:
        return None
    if pruning == PRUNING_IREP:
        return IREP()
    raise ValueError('Invalid value given for parameter \'pruning\': ' + str(pruning))


def _create_stopping_criteria(num_rules: int, time_limit: int) -> List[StoppingCriterion]:
    stopping_criteria: List[StoppingCriterion] = []

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

    return stopping_criteria


class MLRuleLearner(MLLearner):
    """
    A scikit-multilearn implementation of a rule learning algorithm for multi-label classification or ranking.

    Attributes
        stats_          Statistics about the training data set
        theory_         The theory that contains the classification rules
        persistence     The 'ModelPersistence' to be used to load/save the theory
    """

    def __init__(self, model_dir: str):
        super().__init__(model_dir)
        self.require_dense = [True, True]  # We need a dense representation of the training data

    def get_model_prefix(self) -> str:
        return 'rules'

    def _fit(self, stats: Stats, x: np.ndarray, y: np.ndarray, random_state: int):
        # Create a dense representation of the training data
        x = self._ensure_input_format(x)
        y = self._ensure_input_format(y)

        # Induce rules
        rule_induction = self._create_rule_induction(stats)
        rule_induction.random_state = random_state
        theory = rule_induction.induce_rules(stats, x, y)
        return theory

    def _predict(self, model, stats: Stats, x: np.ndarray, random_state: int) -> np.ndarray:
        # Create a dense representation of the given examples
        x = self._ensure_input_format(x)

        # Convert feature matrix into Fortran-contiguous array
        x = np.asfortranarray(x, dtype=DTYPE_FLOAT32)

        prediction = self._create_prediction()
        prediction.random_state = self.random_state
        return prediction.predict(stats, model, x)

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


class Boomer(MLRuleLearner):
    """
    A scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
    classification rules.
    """

    def __init__(self, model_dir: str = None, num_rules: int = 1000, time_limit: int = -1, head_refinement: str = None,
                 loss: str = LOSS_LABEL_WISE_LOGISTIC, label_sub_sampling: int = -1,
                 instance_sub_sampling: str = INSTANCE_SUB_SAMPLING_BAGGING,
                 feature_sub_sampling: str = FEATURE_SUB_SAMPLING_RANDOM,
                 pruning: str = None, shrinkage: float = 0.3, l2_regularization_weight: float = 1.0):
        """
        :param num_rules:                   The number of rules to be induced (including the default rule)
        :param time_limit:                  The duration in seconds after which the induction of rules should be
                                            canceled
        :param head_refinement:             The strategy that is used to find the heads of rules. Must be
                                            `single-label`, `full` or None, if the default strategy should be used
        :param loss:                        The loss function to be minimized. Must be `label-wise-squared-error-loss`,
                                            `label-wise-logistic-loss` or `example-wise-logistic-loss`
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

    def get_model_prefix(self) -> str:
        return 'boomer'

    def get_name(self) -> str:
        name = 'num-rules=' + str(self.num_rules)
        if self.head_refinement is not None:
            name += '_head-refinement=' + str(self.head_refinement)
        name += '_loss=' + str(self.loss)
        if self.label_sub_sampling is not None:
            name += '_label-sub-sampling=' + str(self.label_sub_sampling)
        if self.instance_sub_sampling is not None:
            name += '_instance-sub-sampling=' + str(self.instance_sub_sampling)
        if self.feature_sub_sampling is not None:
            name += '_feature-sub-sampling=' + str(self.feature_sub_sampling)
        if self.pruning is not None:
            name += '_pruning=' + str(self.pruning)
        if 0.0 < float(self.shrinkage) < 1.0:
            name += '_shrinkage=' + str(self.shrinkage)
        if float(self.l2_regularization_weight) > 0.0:
            name += '_l2=' + str(self.l2_regularization_weight)
        return name

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

    def _create_prediction(self) -> Prediction:
        return Sign(LinearCombination())

    def _create_rule_induction(self, stats: Stats) -> RuleInduction:
        stopping_criteria = _create_stopping_criteria(int(self.num_rules), int(self.time_limit))
        l2_regularization_weight = self.__create_l2_regularization_weight()
        loss = self.__create_loss(l2_regularization_weight)
        head_refinement = self.__create_head_refinement(loss)
        label_sub_sampling = _create_label_sub_sampling(int(self.label_sub_sampling), stats)
        instance_sub_sampling = _create_instance_sub_sampling(str(self.instance_sub_sampling))
        feature_sub_sampling = _create_feature_sub_sampling(str(self.feature_sub_sampling))
        pruning = _create_pruning(str(self.pruning))
        shrinkage = self.__create_shrinkage()
        return GradientBoosting(head_refinement, loss, label_sub_sampling, instance_sub_sampling, feature_sub_sampling,
                                pruning, shrinkage, *stopping_criteria)

    def __create_l2_regularization_weight(self) -> float:
        l2_regularization_weight = float(self.l2_regularization_weight)

        if l2_regularization_weight < 0:
            raise ValueError(
                'Invalid value given for parameter \'l2_regularization_weight\': ' + str(l2_regularization_weight))

        return l2_regularization_weight

    def __create_loss(self, l2_regularization_weight: float) -> Loss:
        loss = self.loss

        if loss == LOSS_LABEL_WISE_SQUARED_ERROR:
            return LabelWiseSquaredErrorLoss(l2_regularization_weight)
        elif loss == LOSS_LABEL_WISE_LOGISTIC:
            return LabelWiseLogisticLoss(l2_regularization_weight)
        elif loss == LOSS_EXAMPLE_WISE_LOGISTIC:
            return ExampleWiseLogisticLoss(l2_regularization_weight)
        raise ValueError('Invalid value given for parameter \'loss\': ' + str(loss))

    def __create_head_refinement(self, loss: Loss) -> HeadRefinement:
        head_refinement = self.head_refinement

        if head_refinement is None:
            return SingleLabelHeadRefinement() if isinstance(loss, DecomposableLoss) else FullHeadRefinement()
        elif head_refinement == HEAD_REFINEMENT_SINGLE:
            return SingleLabelHeadRefinement()
        elif head_refinement == HEAD_REFINEMENT_FULL:
            return FullHeadRefinement()
        raise ValueError('Invalid value given for parameter \'head_refinement\': ' + str(head_refinement))

    def __create_shrinkage(self) -> Shrinkage:
        shrinkage = float(self.shrinkage)

        if 0.0 < shrinkage < 1.0:
            return ConstantShrinkage(shrinkage)
        if shrinkage == 1.0:
            return None
        raise ValueError('Invalid value given for parameter \'shrinkage\': ' + str(shrinkage))


class SeparateAndConquerRuleLearner(MLRuleLearner):
    """
    A scikit-multilearn implementation of an Separate-and-Conquer algorithm for learning multi-label
    classification rules.
    """

    def __init__(self, model_dir: str = None, num_rules: int = 500, time_limit: int = -1, head_refinement: str = None,
                 loss: str = MEASURE_LABEL_WISE, heuristic: str = HEURISTIC_PRECISION, label_sub_sampling: int = -1,
                 instance_sub_sampling: str = None, feature_sub_sampling: str = None, pruning: str = None):
        """
        :param num_rules:                   The maximum number of rules to be induced (including the default rule)
        :param time_limit:                  The duration in seconds after which the induction of rules should be
                                            canceled
        :param head_refinement:             The strategy that is used to find the heads of rules. Must be
                                            `single-label` or None, if the default strategy should be used
        :param loss:                        The loss function to be minimized. Must be `label-wise-measure`
        :param heuristic:                   The heuristic to be minimized. Must be `precision` or `hamming-loss`
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
        """
        super().__init__(model_dir)
        self.num_rules = num_rules
        self.time_limit = time_limit
        self.head_refinement = head_refinement
        self.loss = loss
        self.heuristic = heuristic
        self.label_sub_sampling = label_sub_sampling
        self.instance_sub_sampling = instance_sub_sampling
        self.feature_sub_sampling = feature_sub_sampling
        self.pruning = pruning

    def get_model_prefix(self) -> str:
        return 'seco'

    def get_name(self) -> str:
        name = 'num-rules=' + str(self.num_rules)
        if self.head_refinement is not None:
            name += '_head-refinement=' + str(self.head_refinement)
        name += '_loss=' + str(self.loss)
        name += '_heuristic=' + str(self.heuristic)
        if self.label_sub_sampling is not None:
            name += '_label-sub-sampling=' + str(self.label_sub_sampling)
        if self.instance_sub_sampling is not None:
            name += '_instance-sub-sampling=' + str(self.instance_sub_sampling)
        if self.feature_sub_sampling is not None:
            name += '_feature-sub-sampling=' + str(self.feature_sub_sampling)
        if self.pruning is not None:
            name += '_pruning=' + str(self.pruning)
        return name

    def get_params(self, deep=True):
        params = super().get_params()
        params.update({
            'num_rules': self.num_rules,
            'time_limit': self.time_limit,
            'head_refinement': self.head_refinement,
            'loss': self.loss,
            'heuristic': self.heuristic,
            'label_sub_sampling': self.label_sub_sampling,
            'instance_sub_sampling': self.instance_sub_sampling,
            'feature_sub_sampling': self.feature_sub_sampling,
            'pruning': self.pruning
        })
        return params

    def _create_rule_induction(self, stats: Stats) -> RuleInduction:
        heuristic = self.__create_heuristic()
        loss = self.__create_loss(heuristic)
        head_refinement = self.__create_head_refinement()
        label_sub_sampling = _create_label_sub_sampling(int(self.label_sub_sampling), stats)
        instance_sub_sampling = _create_instance_sub_sampling(str(self.instance_sub_sampling))
        feature_sub_sampling = _create_feature_sub_sampling(str(self.feature_sub_sampling))
        pruning = _create_pruning(str(self.pruning))
        stopping_criteria = _create_stopping_criteria(int(self.num_rules), int(self.time_limit))
        stopping_criteria.append(UncoveredLabelsCriterion(loss, 0))
        return SeparateAndConquer(head_refinement, loss, label_sub_sampling, instance_sub_sampling,
                                  feature_sub_sampling, pruning, *stopping_criteria)

    def __create_heuristic(self) -> Heuristic:
        heuristic = self.heuristic

        if heuristic == HEURISTIC_PRECISION:
            return Precision()
        elif heuristic == HEURISTIC_HAMMING_LOSS:
            return HammingLoss()
        raise ValueError('Invalid value given for parameter \'heuristic\': ' + str(heuristic))

    def __create_loss(self, heuristic: Heuristic) -> Loss:
        loss = self.loss

        if loss == MEASURE_LABEL_WISE:
            return LabelWiseMeasure(heuristic)
        raise ValueError('Invalid value given for parameter \'loss\': ' + str(loss))

    def __create_head_refinement(self) -> HeadRefinement:
        head_refinement = self.head_refinement

        if head_refinement is None:
            return SingleLabelHeadRefinement()
        elif head_refinement == HEAD_REFINEMENT_SINGLE:
            return SingleLabelHeadRefinement()
        raise ValueError('Invalid value given for parameter \'head_refinement\': ' + str(head_refinement))

    def _create_prediction(self) -> Prediction:
        return DecisionList()
