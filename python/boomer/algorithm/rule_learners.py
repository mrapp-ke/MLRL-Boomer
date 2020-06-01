#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides a scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
classification rules. The classifier is composed of several modules, e.g., for rule induction and prediction.
"""
from abc import abstractmethod
from typing import List

import numpy as np
from boomer.algorithm.coverage_losses import CoverageLoss
from boomer.algorithm.differentiable_losses import DifferentiableLoss, DecomposableDifferentiableLoss
from boomer.algorithm.example_wise_losses import ExampleWiseLogisticLoss
from boomer.algorithm.head_refinement import SingleLabelHeadRefinement, FullHeadRefinement, HeadRefinement, \
    PartialHeadRefinement
from boomer.algorithm.heuristics import HammingLoss, Precision, Heuristic
from boomer.algorithm.label_wise_averaging import LabelWiseAveraging
from boomer.algorithm.label_wise_losses import LabelWiseSquaredErrorLoss, LabelWiseLogisticLoss
from boomer.algorithm.losses import Loss
from boomer.algorithm.prediction import Predictor, DensePredictor, Aggregation, SignFunction
from boomer.algorithm.pruning import IREP, Pruning
from boomer.algorithm.rule_induction import ExactGreedyRuleInduction
from boomer.algorithm.sequential_rule_induction import SequentialRuleInduction, RuleListInduction
from boomer.algorithm.shrinkage import ConstantShrinkage, Shrinkage
from boomer.algorithm.lift_functions import LiftFunction, PeakLiftFunction
from boomer.algorithm.stopping_criteria import StoppingCriterion, SizeStoppingCriterion, TimeStoppingCriterion, \
    UncoveredLabelsCriterion
from boomer.algorithm.sub_sampling import FeatureSubSampling, RandomFeatureSubsetSelection
from boomer.algorithm.sub_sampling import InstanceSubSampling, Bagging, RandomInstanceSubsetSelection
from boomer.algorithm.sub_sampling import LabelSubSampling, RandomLabelSubsetSelection
from scipy.sparse import issparse, isspmatrix_lil, isspmatrix_coo, isspmatrix_dok, isspmatrix_csc, isspmatrix_csr
from boomer.algorithm.model import DTYPE_UINT8, DTYPE_INTP, DTYPE_FLOAT32
from boomer.learners import MLLearner, NominalAttributeLearner
from boomer.stats import Stats

HEAD_REFINEMENT_SINGLE = 'single-label'

HEAD_REFINEMENT_PARTIAL = 'partial'

HEAD_REFINEMENT_FULL = 'full'

LOSS_LABEL_WISE_LOGISTIC = 'label-wise-logistic-loss'

LOSS_LABEL_WISE_SQUARED_ERROR = 'label-wise-squared-error-loss'

LOSS_EXAMPLE_WISE_LOGISTIC = 'example-wise-logistic-loss'

AVERAGING_LABEL_WISE = 'label-wise-averaging'

HEURISTIC_PRECISION = 'precision'

HEURISTIC_HAMMING_LOSS = 'hamming-loss'

LABEL_SUB_SAMPLING_RANDOM = 'random-label-selection'

INSTANCE_SUB_SAMPLING_RANDOM = 'random-instance-selection'

INSTANCE_SUB_SAMPLING_BAGGING = 'bagging'

FEATURE_SUB_SAMPLING_RANDOM = 'random-feature-selection'

PRUNING_IREP = 'irep'

LIFT_FUNCTION_PEAK = 'peak'


def _create_label_sub_sampling(label_sub_sampling: str, num_samples: int, stats: Stats) -> LabelSubSampling:
    if label_sub_sampling is None:
        return None
    else:
        if num_samples < 1 or num_samples >= stats.num_labels:
            raise ValueError('Value given for parameter \'label_sub_sampling_num_samples\' (' + str(
                num_samples) + ') must be at least 1 and less than the number of labels in the data set (' + str(
                stats.num_labels) + ')')

        if label_sub_sampling == LABEL_SUB_SAMPLING_RANDOM:
            return RandomLabelSubsetSelection(num_samples)
        raise ValueError('Invalid value given for parameter \'label_sub_sampling\': ' + str(label_sub_sampling))


def _create_instance_sub_sampling(instance_sub_sampling: str, sample_size: float) -> InstanceSubSampling:
    if instance_sub_sampling is None:
        return None
    else:
        if sample_size < 0 or sample_size >= 1:
            raise ValueError(
                'Invalid value given for parameter \'instance_sub_sampling_sample_size\': ' + str(sample_size))

        if instance_sub_sampling == INSTANCE_SUB_SAMPLING_BAGGING:
            return Bagging(sample_size if sample_size > 0 else 1.0)
        elif instance_sub_sampling == INSTANCE_SUB_SAMPLING_RANDOM:
            return RandomInstanceSubsetSelection(sample_size if sample_size > 0 else 0.66)
        raise ValueError('Invalid value given for parameter \'instance_sub_sampling\': ' + str(instance_sub_sampling))


def _create_feature_sub_sampling(feature_sub_sampling: str, sample_size: float) -> FeatureSubSampling:
    if feature_sub_sampling is None:
        return None
    else:
        if sample_size < 0 or sample_size >= 1:
            raise ValueError(
                'Invalid value given for parameter \'feature_sub_sampling_sample_size\': ' + str(sample_size))

        if feature_sub_sampling == FEATURE_SUB_SAMPLING_RANDOM:
            return RandomFeatureSubsetSelection(sample_size)
        raise ValueError('Invalid value given for parameter \'feature_sub_sampling\': ' + str(feature_sub_sampling))


def _create_pruning(pruning: str) -> Pruning:
    if pruning is None:
        return None
    if pruning == PRUNING_IREP:
        return IREP()
    raise ValueError('Invalid value given for parameter \'pruning\': ' + str(pruning))


def _create_stopping_criteria(max_rules: int, time_limit: int) -> List[StoppingCriterion]:
    stopping_criteria: List[StoppingCriterion] = []

    if max_rules != -1:
        if max_rules > 0:
            stopping_criteria.append(SizeStoppingCriterion(max_rules))
        else:
            raise ValueError('Invalid value given for parameter \'max_rules\': ' + str(max_rules))

    if time_limit != -1:
        if time_limit > 0:
            stopping_criteria.append(TimeStoppingCriterion(time_limit))
        else:
            raise ValueError('Invalid value given for parameter \'time_limit\': ' + str(time_limit))

    return stopping_criteria


def _create_min_coverage(min_coverage: int) -> int:
    if min_coverage < 1:
        raise ValueError('Invalid value given for parameter \'min_coverage\':' + str(min_coverage))

    return min_coverage


def _create_max_conditions(max_conditions: int) -> int:
    if max_conditions != -1 and max_conditions < 1:
        raise ValueError('Invalid value given for parameter \'max_conditions\'' + str(max_conditions))

    return max_conditions


class MLRuleLearner(MLLearner, NominalAttributeLearner):
    """
    A scikit-multilearn implementation of a rule learning algorithm for multi-label classification or ranking.
    """

    def __init__(self, model_dir: str):
        super().__init__(model_dir)
        # By default, we use dense feature matrices (first value) and label matrices (second value)
        self.require_dense = [True, True]

    def get_model_prefix(self) -> str:
        return 'rules'

    def _fit(self, stats: Stats, x, y, random_state: int):
        x, y = self._validate_data(x, y, accept_sparse=True, multi_output=True)

        # Create a dense representation of the training data
        x = self._ensure_input_format(x)
        y = self._ensure_output_format(y)

        # Create an array that contains the indices of all nominal attributes, if any
        nominal_attribute_indices = self.nominal_attribute_indices

        if nominal_attribute_indices is not None and len(nominal_attribute_indices) > 0:
            nominal_attribute_indices = np.ascontiguousarray(nominal_attribute_indices, dtype=DTYPE_INTP)
        else:
            nominal_attribute_indices = None

        # Convert feature and label matrices into Fortran-contiguous arrays
        x = np.asfortranarray(x, dtype=DTYPE_FLOAT32)
        y = np.asfortranarray(y, dtype=DTYPE_UINT8)

        # Induce rules
        sequential_rule_induction = self._create_sequential_rule_induction(stats)
        return sequential_rule_induction.induce_rules(nominal_attribute_indices, x, y, random_state)

    def _predict(self, model, stats: Stats, x, random_state: int):
        x = self._validate_data(x, reset=False, accept_sparse=True)
        sparse_format = 'csr'
        enforce_sparse = MLRuleLearner.__should_enforce_sparse(x, sparse_format=sparse_format)
        x = self._ensure_input_format(x, enforce_sparse=enforce_sparse, sparse_format=sparse_format)
        num_labels = stats.num_labels
        predictor = self._create_predictor()

        if issparse(x):
            x_data = np.ascontiguousarray(x.data, dtype=DTYPE_FLOAT32)
            x_row_indices = np.ascontiguousarray(x.indptr, dtype=DTYPE_INTP)
            x_col_indices = np.ascontiguousarray(x.indices, dtype=DTYPE_INTP)
            num_features = x.shape[1]
            return predictor.predict_csr(x_data, x_row_indices, x_col_indices, num_features, num_labels, model)
        else:
            x = np.ascontiguousarray(self._ensure_input_format(x), dtype=DTYPE_FLOAT32)
            return predictor.predict(x, num_labels, model)

    @staticmethod
    def __should_enforce_sparse(m, sparse_format: str = 'csr') -> bool:
        """
        Returns whether it is preferable to convert a given matrix into a `scipy.sparse.csr_matrix` or
        `scipy.sparse.csc_matrix`, depending on the format of the given matrix and on how much memory the sparse matrix
        will occupy compared to a dense matrix. To be able to convert the matrix into a sparse format, it must be a
        `scipy.sparse.lil_matrix`, `scipy.sparse.dok_matrix` or `scipy.sparse.coo_matrix`. If the given matrix is
        already in the specified sparse format or if it is a dense matrix, it will not be converted.

        :param m:               The np.ndarray or scipy.sparse.matrix to be checked
        :param sparse_format:   The sparse format to be used. Must be 'csr' or 'csc'
        :return:                True, if it is preferable to convert the matrix into a sparse matrix of the given
                                format, False otherwise
        """
        if not issparse(m):
            # Given matrix is dense
            return False
        elif (isspmatrix_csr(m) and sparse_format == 'csr') or (isspmatrix_csc(m) and sparse_format == 'csc'):
            # Given matrix is already in the specified sparse format
            return True
        elif isspmatrix_lil(m) or isspmatrix_coo(m) or isspmatrix_dok(m):
            # Given matrix is in a format that might be converted into the specified sparse format
            num_non_zero = m.nnz
            num_pointers = m.shape[1 if sparse_format == 'csc' else 0]
            size_int = np.dtype(DTYPE_INTP).itemsize
            size_float = np.dtype(DTYPE_FLOAT32).itemsize
            size_sparse = (num_non_zero * size_float) + (num_non_zero * size_int) + (num_pointers * size_int)
            size_dense = np.prod(m.shape) * size_float
            return size_sparse < size_dense
        else:
            raise ValueError('Unsupported type of matrix given: ' + type(m).__name__)

    @abstractmethod
    def _create_predictor(self) -> Predictor:
        """
        Must be implemented by subclasses in order to create the `Predictor` to be used for making predictions.

        :return: The `Predictor` that has been created
        """
        pass

    @abstractmethod
    def _create_sequential_rule_induction(self, stats: Stats) -> SequentialRuleInduction:
        """
        Must be implemented by subclasses in order to create the algorithm that should be used for sequential rule
        induction.

        :param stats:   Statistics about the training data set
        :return:        The algorithm for sequential rule induction that has been created
        """
        pass


class Boomer(MLRuleLearner):
    """
    A scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
    classification rules.
    """

    def __init__(self, model_dir: str = None, max_rules: int = 1000, time_limit: int = -1, head_refinement: str = None,
                 loss: str = LOSS_LABEL_WISE_LOGISTIC, label_sub_sampling: str = None,
                 label_sub_sampling_num_samples: int = 1, instance_sub_sampling: str = INSTANCE_SUB_SAMPLING_BAGGING,
                 instance_sub_sampling_sample_size: float = 0.0,
                 feature_sub_sampling: str = FEATURE_SUB_SAMPLING_RANDOM, feature_sub_sampling_sample_size: float = 0.0,
                 pruning: str = None, shrinkage: float = 0.3, l2_regularization_weight: float = 1.0,
                 min_coverage: int = 1, max_conditions: int = -1):
        """
        :param max_rules:                           The maximum number of rules to be induced (including the default
                                                    rule)
        :param time_limit:                          The duration in seconds after which the induction of rules should be
                                                    canceled
        :param head_refinement:                     The strategy that is used to find the heads of rules. Must be
                                                    `single-label`, `full` or None, if the default strategy should be
                                                    used
        :param loss:                                The loss function to be minimized. Must be
                                                    `label-wise-squared-error-loss`, `label-wise-logistic-loss` or
                                                    `example-wise-logistic-loss`
        :param label_sub_sampling:                  The strategy that is used for sub-sampling the labels each time a
                                                    new classification rule is learned. Must be 'random-label-selection'
                                                    or None, if no sub-sampling should be used
        :param label_sub_sampling_num_samples:      The number of samples to be used for sub-sampling the labels. Must
                                                    be at least 1
        :param instance_sub_sampling:               The strategy that is used for sub-sampling the training examples
                                                    each time a new classification rule is learned. Must be `bagging`,
                                                    `random-instance-selection` or None, if no sub-sampling should be
                                                    used
        :param instance_sub_sampling_sample_size:   The fraction of examples to be included when sub-sampling the
                                                    training examples. Must be in (0, 1) or 0, if the default value
                                                    should be used
        :param feature_sub_sampling:                The strategy that is used for sub-sampling the features each time a
                                                    classification rule is refined. Must be `random-feature-selection`
                                                    or None, if no sub-sampling should be used
        :param feature_sub_sampling_sample_size:    The fraction of features to be included when sub-sampling the
                                                    features. Must be in (0, 1) or 0, if the default value should be
                                                    used
        :param pruning:                             The strategy that is used for pruning rules. Must be `irep` or None,
                                                    if no pruning should be used
        :param shrinkage:                           The shrinkage parameter that should be applied to the predictions of
                                                    newly induced rules to reduce their effect on the entire model. Must
                                                    be in (0, 1]
        :param l2_regularization_weight:            The weight of the L2 regularization that is applied for calculating
                                                    the scores that are predicted by rules. Must be at least 0
        :param min_coverage:                        The minimum number of training examples that must be covered by a
                                                    rule. Must be at least 1
        :param max_conditions:                      The maximum number of conditions to be included in a rule's body.
                                                    Must be at least 1 or -1, if the number of conditions should not be
                                                    restricted
        """
        super().__init__(model_dir)
        self.max_rules = max_rules
        self.time_limit = time_limit
        self.head_refinement = head_refinement
        self.loss = loss
        self.label_sub_sampling = label_sub_sampling
        self.label_sub_sampling_num_samples = label_sub_sampling_num_samples
        self.instance_sub_sampling = instance_sub_sampling
        self.instance_sub_sampling_sample_size = instance_sub_sampling_sample_size
        self.feature_sub_sampling = feature_sub_sampling
        self.feature_sub_sampling_sample_size = feature_sub_sampling_sample_size
        self.pruning = pruning
        self.shrinkage = shrinkage
        self.l2_regularization_weight = l2_regularization_weight
        self.min_coverage = min_coverage
        self.max_conditions = max_conditions

    def get_model_prefix(self) -> str:
        return 'boomer'

    def get_name(self) -> str:
        name = 'max-rules=' + str(self.max_rules)
        if self.head_refinement is not None:
            name += '_head-refinement=' + str(self.head_refinement)
        name += '_loss=' + str(self.loss)
        if self.label_sub_sampling is not None:
            name += '_label-sub-sampling=' + str(self.label_sub_sampling)
            name += '_label-sub-sampling-num-samples=' + str(self.label_sub_sampling_num_samples)
        if self.instance_sub_sampling is not None:
            name += '_instance-sub-sampling=' + str(self.instance_sub_sampling)
            if int(self.instance_sub_sampling_sample_size) > 0:
                name += '_instance-sub-sampling-sample-size=' + str(self.instance_sub_sampling_sample_size)
        if self.feature_sub_sampling is not None:
            name += '_feature-sub-sampling=' + str(self.feature_sub_sampling)
            if int(self.feature_sub_sampling_sample_size) > 0:
                name += '_feature_sub_sampling=' + str(self.feature_sub_sampling_sample_size)
        if self.pruning is not None:
            name += '_pruning=' + str(self.pruning)
        if 0.0 < float(self.shrinkage) < 1.0:
            name += '_shrinkage=' + str(self.shrinkage)
        if float(self.l2_regularization_weight) > 0.0:
            name += '_l2=' + str(self.l2_regularization_weight)
        if int(self.min_coverage) > 1:
            name += '_min-coverage=' + str(self.min_coverage)
        if int(self.max_conditions) != -1:
            name += '_max-conditions=' + str(self.max_conditions)
        return name

    def get_params(self, deep=True):
        params = super().get_params()
        params.update({
            'max_rules': self.max_rules,
            'time_limit': self.time_limit,
            'head_refinement': self.head_refinement,
            'loss': self.loss,
            'label_sub_sampling': self.label_sub_sampling,
            'label_sub_sampling_num_samples': self.label_sub_sampling_num_samples,
            'instance_sub_sampling': self.instance_sub_sampling,
            'instance_sub_sampling_sample_size': self.instance_sub_sampling_sample_size,
            'feature_sub_sampling': self.feature_sub_sampling,
            'feature_sub_sampling_sample_size': self.feature_sub_sampling_sample_size,
            'pruning': self.pruning,
            'shrinkage': self.shrinkage,
            'l2_regularization_weight': self.l2_regularization_weight,
            'min_coverage': self.min_coverage,
            'max_conditions': self.max_conditions
        })
        return params

    def _create_predictor(self) -> Predictor:
        return DensePredictor(Aggregation(), SignFunction())

    def _create_sequential_rule_induction(self, stats: Stats) -> SequentialRuleInduction:
        rule_induction = ExactGreedyRuleInduction()
        l2_regularization_weight = self.__create_l2_regularization_weight()
        loss = self.__create_loss(l2_regularization_weight)
        head_refinement = self.__create_head_refinement(loss)
        stopping_criteria = _create_stopping_criteria(int(self.max_rules), int(self.time_limit))
        label_sub_sampling = _create_label_sub_sampling(self.label_sub_sampling,
                                                        int(self.label_sub_sampling_num_samples), stats)
        instance_sub_sampling = _create_instance_sub_sampling(self.instance_sub_sampling,
                                                              int(self.instance_sub_sampling_sample_size))
        feature_sub_sampling = _create_feature_sub_sampling(self.feature_sub_sampling,
                                                            int(self.feature_sub_sampling_sample_size))
        pruning = _create_pruning(self.pruning)
        shrinkage = self.__create_shrinkage()
        min_coverage = _create_min_coverage(self.min_coverage)
        max_conditions = _create_max_conditions(self.max_conditions)
        return RuleListInduction(False, rule_induction, head_refinement, loss, stopping_criteria, label_sub_sampling,
                                 instance_sub_sampling, feature_sub_sampling, pruning, shrinkage, min_coverage,
                                 max_conditions)

    def __create_l2_regularization_weight(self) -> float:
        l2_regularization_weight = float(self.l2_regularization_weight)

        if l2_regularization_weight < 0:
            raise ValueError(
                'Invalid value given for parameter \'l2_regularization_weight\': ' + str(l2_regularization_weight))

        return l2_regularization_weight

    def __create_loss(self, l2_regularization_weight: float) -> DifferentiableLoss:
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
            if isinstance(loss, DecomposableDifferentiableLoss):
                return SingleLabelHeadRefinement()
            else:
                return FullHeadRefinement()
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

    def __init__(self, model_dir: str = None, max_rules: int = 500, time_limit: int = -1, head_refinement: str = None,
                 lift_function: str = LIFT_FUNCTION_PEAK, loss: str = AVERAGING_LABEL_WISE,
                 heuristic: str = HEURISTIC_PRECISION, label_sub_sampling: str = None,
                 label_sub_sampling_num_samples: int = 1, instance_sub_sampling: str = None,
                 instance_sub_sampling_sample_size: float = 0.0, feature_sub_sampling: str = None,
                 feature_sub_sampling_sample_size: float = 0.0, pruning: str = None, min_coverage: int = 1,
                 max_conditions: int = -1):
        """
        :param max_rules:                           The maximum number of rules to be induced (including the default
                                                    rule)
        :param time_limit:                          The duration in seconds after which the induction of rules should be
                                                    canceled
        :param head_refinement:                     The strategy that is used to find the heads of rules. Must be
                                                    `single-label`, `partial` or None, if the default strategy should be used
        :param lift_function:               The lift function to use. Must be `peak`
        :param loss:                                The loss function to be minimized. Must be `label-wise-averaging`
        :param heuristic:                           The heuristic to be minimized. Must be `precision` or `hamming-loss`
        :param label_sub_sampling:                  The strategy that is used for sub-sampling the labels each time a
                                                    new classification rule is learned. Must be 'random-label-selection'
                                                    or None, if no sub-sampling should be used
        :param label_sub_sampling_num_samples:      The number of samples to be used for sub-sampling the labels. Must
                                                    be at least 1
        :param instance_sub_sampling:               The strategy that is used for sub-sampling the training examples
                                                    each time a new classification rule is learned. Must be `bagging`,
                                                    `random-instance-selection` or None, if no sub-sampling should be
                                                    used
        :param instance_sub_sampling_sample_size:   The fraction of examples to be included when sub-sampling the
                                                    training examples. Must be in (0, 1) or 0, if the default value
                                                    should be used
        :param feature_sub_sampling:                The strategy that is used for sub-sampling the features each time a
                                                    classification rule is refined. Must be `random-feature-selection`
                                                    or None, if no sub-sampling should be used
        :param feature_sub_sampling_sample_size:    The fraction of features to be included when sub-sampling the
                                                    features. Must be in (0, 1) or 0, if the default value should be
                                                    used
        :param pruning:                             The strategy that is used for pruning rules. Must be `irep` or None,
                                                    if no pruning should be used
        :param min_coverage:                        The minimum number of training examples that must be covered by a
                                                    rule. Must be at least 1
        :param max_conditions:                      The maximum number of conditions to be included in a rule's body.
                                                    Must be at least 1 or -1, if the number of conditions should not be
                                                    restricted
        """
        super().__init__(model_dir)
        self.max_rules = max_rules
        self.time_limit = time_limit
        self.head_refinement = head_refinement
        self.lift_function = lift_function
        self.loss = loss
        self.heuristic = heuristic
        self.label_sub_sampling = label_sub_sampling
        self.label_sub_sampling_num_samples = label_sub_sampling_num_samples
        self.instance_sub_sampling = instance_sub_sampling
        self.instance_sub_sampling_sample_size = instance_sub_sampling_sample_size
        self.feature_sub_sampling = feature_sub_sampling
        self.feature_sub_sampling_sample_size = feature_sub_sampling_sample_size
        self.pruning = pruning
        self.min_coverage = min_coverage
        self.max_conditions = max_conditions

    def get_model_prefix(self) -> str:
        return 'seco'

    def get_name(self) -> str:
        name = 'max-rules=' + str(self.max_rules)
        if self.head_refinement is not None:
            name += '_head-refinement=' + str(self.head_refinement)
        name += '_lift-function=' + str(self.lift_function)
        name += '_loss=' + str(self.loss)
        name += '_heuristic=' + str(self.heuristic)
        if self.label_sub_sampling is not None:
            name += '_label-sub-sampling=' + str(self.label_sub_sampling)
            name += '_label-sub-sampling-num-samples=' + str(self.label_sub_sampling_num_samples)
        if self.instance_sub_sampling is not None:
            name += '_instance-sub-sampling=' + str(self.instance_sub_sampling)
            if int(self.instance_sub_sampling_sample_size) > 0:
                name += '_instance-sub-sampling-sample-size=' + str(self.instance_sub_sampling_sample_size)
        if self.feature_sub_sampling is not None:
            name += '_feature-sub-sampling=' + str(self.feature_sub_sampling)
            if int(self.feature_sub_sampling_sample_size) > 0:
                name += '_feature_sub_sampling=' + str(self.feature_sub_sampling_sample_size)
        if self.pruning is not None:
            name += '_pruning=' + str(self.pruning)
        if int(self.min_coverage) > 1:
            name += '_min-coverage=' + str(self.min_coverage)
        if int(self.max_conditions) != -1:
            name += '_max-conditions=' + str(self.max_conditions)
        return name

    def get_params(self, deep=True):
        params = super().get_params()
        params.update({
            'max_rules': self.max_rules,
            'time_limit': self.time_limit,
            'head_refinement': self.head_refinement,
            'lift_function': self.lift_function,
            'loss': self.loss,
            'heuristic': self.heuristic,
            'label_sub_sampling': self.label_sub_sampling,
            'label_sub_sampling_num_samples': self.label_sub_sampling_num_samples,
            'instance_sub_sampling': self.instance_sub_sampling,
            'instance_sub_sampling_sample_size': self.instance_sub_sampling_sample_size,
            'feature_sub_sampling': self.feature_sub_sampling,
            'feature_sub_sampling_sample_size': self.feature_sub_sampling_sample_size,
            'pruning': self.pruning,
            'min_coverage': self.min_coverage,
            'max_conditions': self.max_conditions
        })
        return params

    def _create_sequential_rule_induction(self, stats: Stats) -> SequentialRuleInduction:
        rule_induction = ExactGreedyRuleInduction()
        heuristic = self.__create_heuristic()
        loss = self.__create_loss(heuristic)
        lift_function = self.__create_lift_function(stats)
        head_refinement = self.__create_head_refinement(lift_function)
        label_sub_sampling = _create_label_sub_sampling(self.label_sub_sampling,
                                                        int(self.label_sub_sampling_num_samples), stats)
        instance_sub_sampling = _create_instance_sub_sampling(self.instance_sub_sampling,
                                                              self.instance_sub_sampling_sample_size)
        feature_sub_sampling = _create_feature_sub_sampling(self.feature_sub_sampling,
                                                            int(self.feature_sub_sampling_sample_size))
        pruning = _create_pruning(self.pruning)
        min_coverage = _create_min_coverage(self.min_coverage)
        max_conditions = _create_max_conditions(self.max_conditions)
        stopping_criteria = _create_stopping_criteria(int(self.max_rules), int(self.time_limit))
        stopping_criteria.append(UncoveredLabelsCriterion(loss, 0))
        return RuleListInduction(True, rule_induction, head_refinement, loss, stopping_criteria, label_sub_sampling,
                                 instance_sub_sampling, feature_sub_sampling, pruning, None, min_coverage,
                                 max_conditions)

    def __create_heuristic(self) -> Heuristic:
        heuristic = self.heuristic

        if heuristic == HEURISTIC_PRECISION:
            return Precision()
        elif heuristic == HEURISTIC_HAMMING_LOSS:
            return HammingLoss()
        raise ValueError('Invalid value given for parameter \'heuristic\': ' + str(heuristic))

    def __create_loss(self, heuristic: Heuristic) -> CoverageLoss:
        loss = self.loss

        if loss == AVERAGING_LABEL_WISE:
            return LabelWiseAveraging(heuristic)
        raise ValueError('Invalid value given for parameter \'loss\': ' + str(loss))

    def __create_lift_function(self, stats: Stats) -> LiftFunction:
        lift_function = self.lift_function

        if lift_function == LIFT_FUNCTION_PEAK:
            # TODO: Example configuration, target labels are half number of labels rounded up
            return PeakLiftFunction(stats.num_labels, int(stats.num_labels / 2) + 1, 2.0, 1.0)
        raise ValueError('Invalid value given for parameter \'lift_function\': ' + str(lift_function))

    def __create_head_refinement(self, lift_function: LiftFunction) -> HeadRefinement:
        head_refinement = self.head_refinement

        if head_refinement is None:
            return SingleLabelHeadRefinement()
        elif head_refinement == HEAD_REFINEMENT_SINGLE:
            return SingleLabelHeadRefinement()
        elif head_refinement == HEAD_REFINEMENT_PARTIAL:
            return PartialHeadRefinement(lift_function)
        raise ValueError('Invalid value given for parameter \'head_refinement\': ' + str(head_refinement))

    def _create_predictor(self) -> Predictor:
        return DensePredictor(Aggregation(use_mask=True), SignFunction())
