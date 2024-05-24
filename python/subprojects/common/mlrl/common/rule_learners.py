"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for implementing single- or multi-label rule learning algorithms.
"""
import logging as log

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np

from scipy.sparse import issparse, isspmatrix_coo, isspmatrix_csc, isspmatrix_csr, isspmatrix_dok, isspmatrix_lil
from sklearn.utils import check_array

from mlrl.common.arrays import enforce_2d, enforce_dense
from mlrl.common.cython.feature_info import EqualFeatureInfo, FeatureInfo, MixedFeatureInfo
from mlrl.common.cython.feature_matrix import CContiguousFeatureMatrix, CscFeatureMatrix, CsrFeatureMatrix, \
    FortranContiguousFeatureMatrix, RowWiseFeatureMatrix
from mlrl.common.cython.label_matrix import CContiguousLabelMatrix, CsrLabelMatrix
from mlrl.common.cython.label_space_info import LabelSpaceInfo
from mlrl.common.cython.learner import RuleLearner as RuleLearnerWrapper
from mlrl.common.cython.probability_calibration import JointProbabilityCalibrationModel, \
    MarginalProbabilityCalibrationModel
from mlrl.common.cython.rule_model import RuleModel
from mlrl.common.cython.validation import assert_greater_or_equal
from mlrl.common.data_types import Float32, Uint8, Uint32
from mlrl.common.format import format_enum_values
from mlrl.common.learners import IncrementalLearner, Learner, NominalAttributeLearner, OrdinalAttributeLearner

KWARG_SPARSE_FEATURE_VALUE = 'sparse_feature_value'

KWARG_MAX_RULES = 'max_rules'


class SparsePolicy(Enum):
    """
    Specifies all valid textual representation of policies to be used for converting matrices into sparse or dense
    formats.
    """
    AUTO = 'auto'
    FORCE_SPARSE = 'sparse'
    FORCE_DENSE = 'dense'


class SparseFormat(Enum):
    """
    Specifies all valid textual representations of sparse matrix formats.
    """
    CSC = 'csc'
    CSR = 'csr'


def parse_sparse_policy(parameter_name: str, value: Optional[str]) -> SparsePolicy:
    """
    Parses and returns a parameter value that specifies a `SparsePolicy` to be used for converting matrices into sparse
    or dense formats. If the given value is invalid, a `ValueError` is raised.

    :param parameter_name:  The name of the parameter
    :param value:           The value to be parsed or None, if the default value should be used
    :return:                A `SparsePolicy`
    """
    try:
        if value:
            return SparsePolicy(value)

        return SparsePolicy.AUTO
    except ValueError as error:
        raise ValueError('Invalid value given for parameter "' + parameter_name + '": Must be one of '
                         + format_enum_values(SparsePolicy) + ', but is "' + str(value) + '"') from error


def is_sparse(matrix, sparse_format: SparseFormat, dtype, sparse_values: bool = True) -> bool:
    """
    Returns whether a given matrix is considered sparse or not. A matrix is considered sparse if it is given in a sparse
    format and is expected to occupy less memory than a dense matrix.

    :param matrix:          A `np.ndarray` or `scipy.sparse.matrix` to be checked
    :param sparse_format:   The `SparseFormat` to be used
    :param dtype:           The type of the values that should be stored in the matrix
    :param sparse_values:   True, if the values must explicitly be stored when using a sparse format, False otherwise
    :return:                True, if the given matrix is considered sparse, False otherwise
    """
    if issparse(matrix):
        num_pointers = matrix.shape[1 if sparse_format == SparseFormat.CSC else 0]
        size_int = np.dtype(Uint32).itemsize
        size_data = np.dtype(dtype).itemsize
        size_sparse_data = size_data if sparse_values else 0
        num_non_zero = matrix.nnz
        size_sparse = (num_non_zero * size_sparse_data) + (num_non_zero * size_int) + (num_pointers * size_int)
        size_dense = np.prod(matrix.shape) * size_data
        return size_sparse < size_dense
    return False


def should_enforce_sparse(matrix,
                          sparse_format: SparseFormat,
                          policy: SparsePolicy,
                          dtype,
                          sparse_values: bool = True) -> bool:
    """
    Returns whether it is preferable to convert a given matrix into a `scipy.sparse.csr_matrix` or
    `scipy.sparse.csc_matrix`, depending on the format of the given matrix and a given `SparsePolicy`:

    If the given policy is `SparsePolicy.AUTO`, the matrix will be converted into the given sparse format, if possible
    and if the sparse matrix is expected to occupy less memory than a dense matrix. To be able to convert the matrix
    into a sparse format, it must be a `scipy.sparse.lil_matrix`, `scipy.sparse.dok_matrix`, `scipy.sparse.coo_matrix`,
    `scipy.sparse.csr_matrix` or `scipy.sparse.csc_matrix`.

    If the given policy is `SparsePolicy.FORCE_SPARSE`, the matrix will always be converted into the specified sparse
    format, if possible.  Dense matrices will never be converted into a sparse format.

    If the given policy is `SparsePolicy.FORCE_DENSE`, the matrix will always be converted into a dense matrix.

    :param matrix:          A `np.ndarray` or `scipy.sparse.matrix` to be checked
    :param sparse_format:   The `SparseFormat` to be used
    :param policy:          The `SparsePolicy` to be used
    :param dtype:           The type of the values that should be stored in the matrix
    :param sparse_values:   True, if the values must explicitly be stored when using a sparse format, False otherwise
    :return:                True, if it is preferable to convert the matrix into a sparse matrix of the given format,
                            False otherwise
    """
    if not issparse(matrix):
        # Given matrix is dense
        return False
    if isspmatrix_lil(matrix) or isspmatrix_coo(matrix) or isspmatrix_dok(matrix) or isspmatrix_csr(
            matrix) or isspmatrix_csc(matrix):
        # Given matrix is in a format that might be converted into the specified sparse format
        if policy == SparsePolicy.AUTO:
            return is_sparse(matrix, sparse_format=sparse_format, dtype=dtype, sparse_values=sparse_values)
        return policy == SparsePolicy.FORCE_SPARSE

    raise ValueError('Matrix of type ' + type(matrix).__name__ + ' cannot be converted to format "' + str(sparse_format)
                     + '"')


def create_binary_predictor(learner: RuleLearnerWrapper, model: RuleModel, label_space_info: LabelSpaceInfo,
                            marginal_probability_calibration_model: MarginalProbabilityCalibrationModel,
                            joint_probability_calibration_model: JointProbabilityCalibrationModel, num_labels: int,
                            feature_matrix: RowWiseFeatureMatrix, sparse: bool):
    """
    Creates and returns a predictor for predicting binary labels.

    :param learner:                                 The learner for which the predictor should be created
    :param model:                                   The model to be used for prediction
    :param label_space_info:                        Information about the label space that may be used for prediction
    :param marginal_probability_calibration_model:  A model for the calibration of marginal probabilities
    :param joint_probability_calibration_model:     A model for the calibration of joint probabilities
    :param num_labels:                              The total number of labels to predict for
    :param feature_matrix:                          A feature matrix that provides row-wise access to the features of
                                                    the query examples
    :param sparse:                                  True, if a sparse matrix should be used for storing predictions,
                                                    False otherwise
    :return:                                        The predictor that has been created
    """
    if sparse:
        return learner.create_sparse_binary_predictor(feature_matrix, model, label_space_info,
                                                      marginal_probability_calibration_model,
                                                      joint_probability_calibration_model, num_labels)
    return learner.create_binary_predictor(feature_matrix, model, label_space_info,
                                           marginal_probability_calibration_model, joint_probability_calibration_model,
                                           num_labels)


def create_score_predictor(learner: RuleLearnerWrapper, model: RuleModel, label_space_info: LabelSpaceInfo,
                           num_labels: int, feature_matrix: RowWiseFeatureMatrix):
    """
    Creates and returns a predictor for predicting regression scores.

    :param learner:             The learner for which the predictor should be created
    :param model:               The model to be used for prediction
    :param label_space_info:    Information about the label space that may be used for prediction
    :param num_labels:          The total number of labels to predict for
    :param feature_matrix:      A feature matrix that provides row-wise access to the features of the query examples
    :return:                    The predictor that has been created
    """
    return learner.create_score_predictor(feature_matrix, model, label_space_info, num_labels)


def create_probability_predictor(learner: RuleLearnerWrapper, model: RuleModel, label_space_info: LabelSpaceInfo,
                                 marginal_probability_calibration_model: MarginalProbabilityCalibrationModel,
                                 joint_probability_calibration_model: JointProbabilityCalibrationModel, num_labels: int,
                                 feature_matrix: RowWiseFeatureMatrix):
    """
    Creates and returns a predictor for predicting probability estimates.

    :param learner:                                 The learner for which the predictor should be created
    :param model:                                   The model to be used for prediction
    :param label_space_info:                        Information about the label space that may be used for prediction
    :param marginal_probability_calibration_model:  A model for the calibration of marginal probabilities
    :param joint_probability_calibration_model:     A model for the calibration of joint probabilities
    :param num_labels:                              The total number of labels to predict for
    :param feature_matrix:                          A feature matrix that provides row-wise access to the features of
                                                    the query examples
    :return:                                        The predictor that has been created
    """
    return learner.create_probability_predictor(feature_matrix, model, label_space_info,
                                                marginal_probability_calibration_model,
                                                joint_probability_calibration_model, num_labels)


def convert_into_sklearn_compatible_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """
    Converts given probability estimates into a format that is compatible with scikit-learn.

    :param probabilities: A `np.ndarray` that stores probability estimates
    :return:              A `np.ndarray` that is compatible with scikit-learn
    """
    if probabilities.shape[1] == 1:
        # In the case of a single-label problem, scikit-learn expects probability estimates to be given for the negative
        # and positive class...
        probabilities = np.hstack((1 - probabilities, probabilities))

    return probabilities


class RuleLearner(Learner, NominalAttributeLearner, OrdinalAttributeLearner, IncrementalLearner, ABC):
    """
    A scikit-learn implementation of a rule learning algorithm for multi-label classification or ranking.
    """

    class NativeIncrementalPredictor(IncrementalLearner.IncrementalPredictor):
        """
        Allows to obtain predictions from a `RuleLearner` incrementally by using its native support of this
        functionality.
        """

        def __init__(self, feature_matrix: RowWiseFeatureMatrix, incremental_predictor):
            """
            :param feature_matrix:          A `RowWiseFeatureMatrix` that stores the feature values of the query
                                            examples
            :param incremental_predictor:   The incremental predictor to be used for obtaining predictions
            """
            self.feature_matrix = feature_matrix
            self.incremental_predictor = incremental_predictor

        def has_next(self) -> bool:
            """
            See :func:`mlrl.common.learners.IncrementalLearner.IncrementalPredictor.has_next`
            """
            return self.incremental_predictor.has_next()

        def get_num_next(self) -> int:
            """
            See :func:`mlrl.common.learners.IncrementalLearner.IncrementalPredictor.get_num_next`
            """
            return self.incremental_predictor.get_num_next()

        def apply_next(self, step_size: int):
            """
            See :func:`mlrl.common.learners.IncrementalLearner.IncrementalPredictor.apply_next`
            """
            return self.incremental_predictor.apply_next(step_size)

    class NativeIncrementalProbabilityPredictor(NativeIncrementalPredictor):
        """
        Allows to obtain probability estimates from a `RuleLearner` incrementally by using its native support of this
        functionality.
        """

        def apply_next(self, step_size: int):
            return convert_into_sklearn_compatible_probabilities(super().apply_next(step_size))

    class IncrementalPredictor(IncrementalLearner.IncrementalPredictor):
        """
        Allows to obtain predictions from a `RuleLearner` incrementally.
        """

        def __init__(self, feature_matrix: RowWiseFeatureMatrix, model: RuleModel, max_rules: int, predictor):
            """
            :param feature_matrix:  A `RowWiseFeatureMatrix` that stores the feature values of the query examples
            :param model:           The model to be used for obtaining predictions
            :param max_rules:       The maximum number of rules to be used for prediction. Must be at least 1 or 0, if
                                    the number of rules should not be restricted
            :param predictor:       The predictor to be used for obtaining predictions
            """
            if max_rules != 0:
                assert_greater_or_equal('max_rules', max_rules, 1)
            self.feature_matrix = feature_matrix
            self.num_total_rules = min(model.get_num_used_rules(),
                                       max_rules) if max_rules > 0 else model.get_num_used_rules()
            self.predictor = predictor
            self.num_considered_rules = 0

        def get_num_next(self) -> int:
            """
            See :func:`mlrl.common.learners.IncrementalLearner.IncrementalPredictor.get_num_next`
            """
            return self.num_total_rules - self.num_considered_rules

        def apply_next(self, step_size: int):
            """
            See :func:`mlrl.common.learners.IncrementalLearner.IncrementalPredictor.apply_next`
            """
            assert_greater_or_equal('step_size', step_size, 1)
            self.num_considered_rules = min(self.num_total_rules, self.num_considered_rules + step_size)
            return self.predictor.predict(self.num_considered_rules)

    class IncrementalProbabilityPredictor(IncrementalPredictor):
        """
        Allows to obtain probability estimates from a `RuleLearner` incrementally.
        """

        def apply_next(self, step_size: int):
            return convert_into_sklearn_compatible_probabilities(super().apply_next(step_size))

    def __init__(self, random_state: Optional[int], feature_format: Optional[str], label_format: Optional[str],
                 prediction_format: Optional[str]):
        """
        :param random_state:        The seed to be used by RNGs. Must be at least 1
        :param feature_format:      The format to be used for the representation of the feature matrix. Must be
                                    `sparse`, `dense` or `auto`
        :param label_format:        The format to be used for the representation of the label matrix. Must be `sparse`,
                                    `dense` or 'auto'
        :param prediction_format:   The format to be used for representation of predicted labels. Must be `sparse`,
                                    `dense` or `auto`
        """
        super().__init__()
        self.random_state = random_state
        self.feature_format = feature_format
        self.label_format = label_format
        self.prediction_format = prediction_format

    # pylint: disable=attribute-defined-outside-init
    def _fit(self, x, y, **kwargs):
        """
        :keyword sparse_feature_value: The value that should be used for sparse elements in the feature matrix. Does
                                       only have an effect if `x` is a `scipy.sparse` matrix
        """
        # Validate feature matrix and convert it to the preferred format...
        sparse_feature_value = float(kwargs.get(KWARG_SPARSE_FEATURE_VALUE, 0.0))
        x_sparse_format = SparseFormat.CSC
        x_sparse_policy = parse_sparse_policy('feature_format', self.feature_format)
        x_enforce_sparse = should_enforce_sparse(x,
                                                 sparse_format=x_sparse_format,
                                                 policy=x_sparse_policy,
                                                 dtype=Float32)
        x = self._validate_data(x if x_enforce_sparse else enforce_2d(
            enforce_dense(x, order='F', dtype=Float32, sparse_value=sparse_feature_value)),
                                accept_sparse=x_sparse_format.value,
                                dtype=Float32,
                                force_all_finite='allow-nan')

        if issparse(x):
            log.debug('A sparse matrix is used to store the feature values of the training examples')
            x_data = np.ascontiguousarray(x.data, dtype=Float32)
            x_indices = np.ascontiguousarray(x.indices, dtype=Uint32)
            x_indptr = np.ascontiguousarray(x.indptr, dtype=Uint32)
            feature_matrix = CscFeatureMatrix(x_data, x_indices, x_indptr, x.shape[0], x.shape[1], sparse_feature_value)
        else:
            log.debug('A dense matrix is used to store the feature values of the training examples')
            feature_matrix = FortranContiguousFeatureMatrix(x)

        # Validate label matrix and convert it to the preferred format...
        y_sparse_format = SparseFormat.CSR
        prediction_sparse_policy = parse_sparse_policy('prediction_format', self.prediction_format)
        self.sparse_predictions_ = prediction_sparse_policy != SparsePolicy.FORCE_DENSE and (
            prediction_sparse_policy == SparsePolicy.FORCE_SPARSE
            or is_sparse(y, sparse_format=y_sparse_format, dtype=Uint8, sparse_values=False))

        y_sparse_policy = parse_sparse_policy('label_format', self.label_format)
        y_enforce_sparse = should_enforce_sparse(y,
                                                 sparse_format=y_sparse_format,
                                                 policy=y_sparse_policy,
                                                 dtype=Uint8,
                                                 sparse_values=False)
        y = check_array(y if y_enforce_sparse else enforce_2d(enforce_dense(y, order='C', dtype=Uint8)),
                        accept_sparse=y_sparse_format.value,
                        dtype=Uint8)

        if issparse(y):
            log.debug('A sparse matrix is used to store the labels of the training examples')
            y_indices = np.ascontiguousarray(y.indices, dtype=Uint32)
            y_indptr = np.ascontiguousarray(y.indptr, dtype=Uint32)
            label_matrix = CsrLabelMatrix(y_indices, y_indptr, y.shape[0], y.shape[1])
        else:
            log.debug('A dense matrix is used to store the labels of the training examples')
            label_matrix = CContiguousLabelMatrix(y)

        # Obtain information about the types of the individual features...
        feature_info = self.__create_feature_info(feature_matrix.get_num_features())

        # Induce rules...
        learner = self._create_learner()
        random_state = int(self.random_state) if self.random_state else 1
        training_result = learner.fit(feature_info, feature_matrix, label_matrix, random_state)
        self.num_labels_ = training_result.num_labels
        self.label_space_info_ = training_result.label_space_info
        self.marginal_probability_calibration_model_ = training_result.marginal_probability_calibration_model
        self.joint_probability_calibration_model_ = training_result.joint_probability_calibration_model
        return training_result.rule_model

    def __create_feature_info(self, num_features: int) -> FeatureInfo:
        """
        Creates and returns a `FeatureInfo` that provides information about the types of individual features.

        :param num_features:    The total number of available features
        :return:                The `FeatureInfo` that has been created
        """
        ordinal_attribute_indices = self.ordinal_attribute_indices
        nominal_attribute_indices = self.nominal_attribute_indices
        num_ordinal_features = 0 if ordinal_attribute_indices is None else len(ordinal_attribute_indices)
        num_nominal_features = 0 if nominal_attribute_indices is None else len(nominal_attribute_indices)

        if num_ordinal_features == 0 and num_nominal_features == 0:
            return EqualFeatureInfo.create_numerical()
        if num_ordinal_features == num_features:
            return EqualFeatureInfo.create_ordinal()
        if num_nominal_features == num_features:
            return EqualFeatureInfo.create_nominal()
        return MixedFeatureInfo(num_features, ordinal_attribute_indices, nominal_attribute_indices)

    def _predict_binary(self, x, **kwargs):
        """
        :keyword sparse_feature_value: The value that should be used for sparse elements in the feature matrix. Does
                                       only have an effect if `x` is a `scipy.sparse` matrix
        """
        learner = self._create_learner()
        feature_matrix = self.__create_row_wise_feature_matrix(x, **kwargs)
        num_labels = self.num_labels_

        if learner.can_predict_binary(feature_matrix, num_labels):
            sparse_predictions = self.sparse_predictions_
            log.debug('A %s matrix is used to store the predicted labels', 'sparse' if sparse_predictions else 'dense')
            max_rules = int(kwargs.get(KWARG_MAX_RULES, 0))
            return create_binary_predictor(learner, self.model_, self.label_space_info_,
                                           self.marginal_probability_calibration_model_,
                                           self.joint_probability_calibration_model_, num_labels, feature_matrix,
                                           sparse_predictions).predict(max_rules)

        return super()._predict_binary(x, **kwargs)

    def _predict_binary_incrementally(self, x, **kwargs):
        """
        :keyword sparse_feature_value:  The value that should be used for sparse elements in the feature matrix. Does
                                        only have an effect if `x` is a `scipy.sparse` matrix
        :keyword max_rules:             The maximum number of rules to be used for prediction. Must be at least 1 or 0,
                                        if the number of rules should not be restricted
        """
        learner = self._create_learner()
        feature_matrix = self.__create_row_wise_feature_matrix(x, **kwargs)
        num_labels = self.num_labels_

        if learner.can_predict_binary(feature_matrix, num_labels):
            sparse_predictions = self.sparse_predictions_
            log.debug('A %s matrix is used to store the predicted labels', 'sparse' if sparse_predictions else 'dense')
            model = self.model_
            predictor = create_binary_predictor(learner, model, self.label_space_info_,
                                                self.marginal_probability_calibration_model_,
                                                self.joint_probability_calibration_model_, num_labels, feature_matrix,
                                                sparse_predictions)
            max_rules = int(kwargs.get(KWARG_MAX_RULES, 0))

            if predictor.can_predict_incrementally():
                return RuleLearner.NativeIncrementalPredictor(feature_matrix,
                                                              predictor.create_incremental_predictor(max_rules))
            return RuleLearner.IncrementalPredictor(feature_matrix, model, max_rules, predictor)

        return super()._predict_binary_incrementally(x, **kwargs)

    def _predict_scores(self, x, **kwargs):
        """
        :keyword sparse_feature_value: The value that should be used for sparse elements in the feature matrix. Does
                                       only have an effect if `x` is a `scipy.sparse` matrix
        """
        learner = self._create_learner()
        feature_matrix = self.__create_row_wise_feature_matrix(x, **kwargs)
        num_labels = self.num_labels_

        if learner.can_predict_scores(feature_matrix, num_labels):
            log.debug('A dense matrix is used to store the predicted regression scores')
            max_rules = int(kwargs.get(KWARG_MAX_RULES, 0))
            return create_score_predictor(learner, self.model_, self.label_space_info_, num_labels,
                                          feature_matrix).predict(max_rules)

        return super()._predict_scores(x, **kwargs)

    def _predict_scores_incrementally(self, x, **kwargs):
        """
        :keyword sparse_feature_value:  The value that should be used for sparse elements in the feature matrix. Does
                                        only have an effect if `x` is a `scipy.sparse` matrix
        :keyword max_rules:             The maximum number of rules to be used for prediction. Must be at least 1 or 0,
                                        if the number of rules should not be restricted
        """
        learner = self._create_learner()
        feature_matrix = self.__create_row_wise_feature_matrix(x, **kwargs)
        num_labels = self.num_labels_

        if learner.can_predict_scores(feature_matrix, num_labels):
            log.debug('A dense matrix is used to store the predicted regression scores')
            model = self.model_
            predictor = create_score_predictor(learner, model, self.label_space_info_, num_labels, feature_matrix)
            max_rules = int(kwargs.get(KWARG_MAX_RULES, 0))

            if predictor.can_predict_incrementally():
                return RuleLearner.NativeIncrementalPredictor(feature_matrix,
                                                              predictor.create_incremental_predictor(max_rules))
            return RuleLearner.IncrementalPredictor(feature_matrix, model, max_rules, predictor)

        return super()._predict_scores_incrementally(x, **kwargs)

    def _predict_proba(self, x, **kwargs):
        """
        :keyword sparse_feature_value: The value that should be used for sparse elements in the feature matrix. Does
                                       only have an effect if `x` is a `scipy.sparse` matrix
        """
        learner = self._create_learner()
        feature_matrix = self.__create_row_wise_feature_matrix(x, **kwargs)
        num_labels = self.num_labels_

        if learner.can_predict_probabilities(feature_matrix, num_labels):
            log.debug('A dense matrix is used to store the predicted probability estimates')
            max_rules = int(kwargs.get(KWARG_MAX_RULES, 0))
            return convert_into_sklearn_compatible_probabilities(
                create_probability_predictor(learner, self.model_, self.label_space_info_,
                                             self.marginal_probability_calibration_model_,
                                             self.joint_probability_calibration_model_, num_labels,
                                             feature_matrix).predict(max_rules))

        return super()._predict_proba(x, **kwargs)

    def _predict_proba_incrementally(self, x, **kwargs):
        """
        :keyword sparse_feature_value:  The value that should be used for sparse elements in the feature matrix. Does
                                        only have an effect if `x` is a `scipy.sparse` matrix
        :keyword max_rules:             The maximum number of rules to be used for prediction. Must be at least 1 or 0,
                                        if the number of rules should not be restricted
        """
        learner = self._create_learner()
        feature_matrix = self.__create_row_wise_feature_matrix(x, **kwargs)
        num_labels = self.num_labels_

        if learner.can_predict_probabilities(feature_matrix, num_labels):
            log.debug('A dense matrix is used to store the predicted probability estimates')
            model = self.model_
            predictor = create_probability_predictor(learner, model, self.label_space_info_,
                                                     self.marginal_probability_calibration_model_,
                                                     self.joint_probability_calibration_model_, num_labels,
                                                     feature_matrix)
            max_rules = int(kwargs.get(KWARG_MAX_RULES, 0))

            if predictor.can_predict_incrementally():
                return RuleLearner.NativeIncrementalProbabilityPredictor(
                    feature_matrix, predictor.create_incremental_predictor(max_rules))
            return RuleLearner.IncrementalProbabilityPredictor(feature_matrix, model, max_rules, predictor)

        return super().predict_proba_incrementally(x, **kwargs)

    def __create_row_wise_feature_matrix(self, x, **kwargs) -> RowWiseFeatureMatrix:
        """
        :keyword sparse_feature_value: The value that should be used for sparse elements in the feature matrix. Does
                                       only have an effect if `x` is a `scipy.sparse` matrix
        """
        sparse_feature_value = float(kwargs.get(KWARG_SPARSE_FEATURE_VALUE, 0.0))
        sparse_format = SparseFormat.CSR
        sparse_policy = parse_sparse_policy('feature_format', self.feature_format)
        enforce_sparse = should_enforce_sparse(x, sparse_format=sparse_format, policy=sparse_policy, dtype=Float32)
        x = self._validate_data(x if enforce_sparse else enforce_2d(
            enforce_dense(x, order='C', dtype=Float32, sparse_value=sparse_feature_value)),
                                reset=False,
                                accept_sparse=sparse_format.value,
                                dtype=Float32,
                                force_all_finite='allow-nan')

        if issparse(x):
            log.debug('A sparse matrix is used to store the feature values of the query examples')
            x_data = np.ascontiguousarray(x.data, dtype=Float32)
            x_indices = np.ascontiguousarray(x.indices, dtype=Uint32)
            x_indptr = np.ascontiguousarray(x.indptr, dtype=Uint32)
            return CsrFeatureMatrix(x_data, x_indices, x_indptr, x.shape[0], x.shape[1], sparse_feature_value)

        log.debug('A dense matrix is used to store the feature values of the query examples')
        return CContiguousFeatureMatrix(x)

    @abstractmethod
    def _create_learner(self) -> RuleLearnerWrapper:
        """
        Must be implemented by subclasses in order to configure and create an implementation of the rule learner.

        :return: The implementation of the rule learner that has been created
        """
