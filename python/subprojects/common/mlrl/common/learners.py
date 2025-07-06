"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for implementing rule learning algorithms.
"""
import logging as log

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Optional, Set

import numpy as np

from sklearn.base import BaseEstimator as SkLearnBaseEstimator
from sklearn.utils.validation import check_array, validate_data

from mlrl.common.config.parameters import Parameter
from mlrl.common.cython.example_weights import EqualExampleWeights, ExampleWeights, RealValuedExampleWeights
from mlrl.common.cython.feature_info import EqualFeatureInfo, FeatureInfo, MixedFeatureInfo
from mlrl.common.cython.feature_matrix import CContiguousFeatureMatrix, ColumnWiseFeatureMatrix, CscFeatureMatrix, \
    CsrFeatureMatrix, FortranContiguousFeatureMatrix, RowWiseFeatureMatrix
from mlrl.common.cython.label_matrix import CContiguousLabelMatrix, CsrLabelMatrix
from mlrl.common.cython.learner import RuleLearnerConfig
from mlrl.common.cython.learner_classification import ClassificationRuleLearner as RuleLearnerWrapper
from mlrl.common.cython.output_space_info import OutputSpaceInfo
from mlrl.common.cython.probability_calibration import JointProbabilityCalibrationModel, \
    MarginalProbabilityCalibrationModel
from mlrl.common.cython.regression_matrix import CContiguousRegressionMatrix, CsrRegressionMatrix
from mlrl.common.cython.rule_model import RuleModel
from mlrl.common.mixins import ClassifierMixin, IncrementalClassifierMixin, IncrementalPredictor, \
    IncrementalRegressorMixin, NominalFeatureSupportMixin, OrdinalFeatureSupportMixin, RegressorMixin

from mlrl.util.arrays import SparseFormat, enforce_2d, enforce_dense, is_sparse, is_sparse_and_memory_efficient
from mlrl.util.options import parse_enum
from mlrl.util.validation import assert_greater_or_equal


class SparsePolicy(StrEnum):
    """
    Specifies all valid textual representation of policies to be used for converting matrices into sparse or dense
    formats.
    """
    AUTO = 'auto'
    FORCE_SPARSE = 'sparse'
    FORCE_DENSE = 'dense'

    def should_enforce_sparse(self, matrix, sparse_format: SparseFormat, dtype, sparse_values: bool = True) -> bool:
        """
        Returns whether it is preferable to convert a given matrix into a `scipy.sparse.csr_array` or
        `scipy.sparse.csc_array`, depending on the format of the given matrix and this `SparsePolicy`:

        If the policy is `SparsePolicy.AUTO`, the matrix will be converted into the given sparse format, if possible and
        if the sparse matrix is expected to occupy less memory than a dense matrix. To be able to convert the matrix
        into a sparse format, it must be a `scipy.sparse.spmatrix` or `scipy.sparse.sparray` in the LIL, DOK, COO, CSR
        or CSC format.

        If the policy is `SparsePolicy.FORCE_SPARSE`, the matrix will always be converted into the specified sparse
        format, if possible.  Dense matrices will never be converted into a sparse format.

        If the policy is `SparsePolicy.FORCE_DENSE`, the matrix will always be converted into a dense matrix.

        :param matrix:          A `np.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray` to be checked
        :param sparse_format:   The `SparseFormat` to be used
        :param dtype:           The type of the values that should be stored in the matrix
        :param sparse_values:   True, if the values must explicitly be stored when using a sparse format, False
                                otherwise
        :return:                True, if it is preferable to convert the matrix into a sparse matrix of the given
                                format, False otherwise
        """
        if not is_sparse(matrix):
            # Given matrix is dense
            return False

        supported_formats = [SparseFormat.LIL, SparseFormat.COO, SparseFormat.DOK, SparseFormat.CSR, SparseFormat.CSC]

        if is_sparse(matrix, supported_formats=supported_formats):
            # Given matrix is in a format that might be converted into the specified sparse format
            if self == SparsePolicy.AUTO:
                return is_sparse_and_memory_efficient(matrix,
                                                      sparse_format=sparse_format,
                                                      dtype=dtype,
                                                      sparse_values=sparse_values)
            return self == SparsePolicy.FORCE_SPARSE

        raise ValueError('Matrix of type ' + type(matrix).__name__ + ' cannot be converted to format "'
                         + str(sparse_format) + '"')


class RuleLearner(SkLearnBaseEstimator, NominalFeatureSupportMixin, OrdinalFeatureSupportMixin, ABC):
    """
    A scikit-learn implementation of a rule learning algorithm.
    """

    KWARG_EXAMPLE_WEIGHTS = 'sample_weights'

    KWARG_SPARSE_FEATURE_VALUE = 'sparse_feature_value'

    KWARG_MAX_RULES = 'max_rules'

    class NativeIncrementalPredictor(IncrementalPredictor):
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
            See :func:`mlrl.common.mixins.IncrementalPredictor.has_next`
            """
            return self.incremental_predictor.has_next()

        def get_num_next(self) -> int:
            """
            See :func:`mlrl.common.mixins.IncrementalPredictor.get_num_next`
            """
            return self.incremental_predictor.get_num_next()

        def apply_next(self, step_size: int):
            """
            See :func:`mlrl.common.mixins.IncrementalPredictor.apply_next`
            """
            return self.incremental_predictor.apply_next(step_size)

    class NonNativeIncrementalPredictor(IncrementalPredictor):
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
            See :func:`mlrl.common.mixins.IncrementalPredictor.get_num_next`
            """
            return self.num_total_rules - self.num_considered_rules

        def apply_next(self, step_size: int):
            """
            See :func:`mlrl.common.mixins.IncrementalPredictor.apply_next`
            """
            assert_greater_or_equal('step_size', step_size, 1)
            self.num_considered_rules = min(self.num_total_rules, self.num_considered_rules + step_size)
            return self.predictor.predict(self.num_considered_rules)

    def __init__(self, feature_format: Optional[str], output_format: Optional[str], prediction_format: Optional[str]):
        """
        :param feature_format:      The format to be used for the representation of the feature matrix. Must be
                                    `sparse`, `dense` or `auto`
        :param output_format:       The format to be used for the representation of the output matrix. Must be `sparse`,
                                    `dense` or 'auto'
        :param prediction_format:   The format to be used for the representation of predictions. Must be `sparse`,
                                    `dense` or `auto`
        """
        super().__init__()
        self.feature_format = feature_format
        self.output_format = output_format
        self.prediction_format = prediction_format
        self.model_: Optional[Any] = None

    # pylint: disable=attribute-defined-outside-init
    def _fit(self, x, y, **kwargs):
        """
        :keyword sparse_feature_value:      The value that should be used for sparse elements in the feature matrix.
                                            Does only have an effect if `x` is a `scipy.sparse.spmatrix` or
                                            `scipy.sparse.sparray`
        :keyword nominal_feature_indices:   A `numpy.ndarray`, shape `(num_nominal_features)`, that stores the indices
                                            of all nominal features
        :keyword ordinal_feature_indices:   A `numpy.ndarray`, shape `(num_ordinal_features)`, that stores the indices
                                            of all ordinal features
        :keyword sample_weights:            A `numpy.ndarray`, shape `(num_examples)`, that stores the weights of
                                            individual training examples
        """
        feature_matrix = self._create_column_wise_feature_matrix(x, **kwargs)
        output_matrix = self.__create_row_wise_output_matrix(y)
        feature_info = self._create_feature_info(feature_matrix.get_num_features(), **kwargs)
        example_weights = self._create_example_weights(feature_matrix.get_num_examples(), **kwargs)
        learner = self._create_learner()
        training_result = learner.fit(example_weights, feature_info, feature_matrix, output_matrix)
        self.num_outputs_ = training_result.num_outputs
        self.output_space_info_ = training_result.output_space_info
        self.marginal_probability_calibration_model_ = training_result.marginal_probability_calibration_model
        self.joint_probability_calibration_model_ = training_result.joint_probability_calibration_model
        return training_result.rule_model

    @staticmethod
    def _create_score_predictor(learner: RuleLearnerWrapper, model: RuleModel, output_space_info: OutputSpaceInfo,
                                num_outputs: int, feature_matrix: RowWiseFeatureMatrix):
        """
        Creates and returns a predictor for predicting scores.

        :param learner:             The learner for which the predictor should be created
        :param model:               The model to be used for prediction
        :param output_space_info:   Information about the output space that may be used for prediction
        :param num_outputs:         The total number of outputs to predict for
        :param feature_matrix:      A feature matrix that provides row-wise access to the features of the query examples
        :return:                    The predictor that has been created
        """
        return learner.create_score_predictor(feature_matrix, model, output_space_info, num_outputs)

    def _predict_scores(self, x, **kwargs):
        """
        :keyword sparse_feature_value: The value that should be used for sparse elements in the feature matrix. Does
                                       only have an effect if `x` is a `scipy.sparse.spmatrix` or `scipy.sparse.sparray`
        """
        learner = self._create_learner()
        feature_matrix = self._create_row_wise_feature_matrix(x, **kwargs)
        num_outputs = self.num_outputs_

        if learner.can_predict_scores(feature_matrix, num_outputs):
            log.debug('A dense matrix is used to store the predicted scores')
            max_rules = int(kwargs.get(self.KWARG_MAX_RULES, 0))
            return self._create_score_predictor(learner, self.model_, self.output_space_info_, num_outputs,
                                                feature_matrix).predict(max_rules)

        raise RuntimeError('Prediction of scores not supported using the current configuration')

    def _predict_scores_incrementally(self, x, **kwargs):
        """
        :keyword sparse_feature_value:  The value that should be used for sparse elements in the feature matrix. Does
                                        only have an effect if `x` is a `scipy.sparse.spmatrix` or
                                        `scipy.sparse.sparray`
        :keyword max_rules:             The maximum number of rules to be used for prediction. Must be at least 1 or 0,
                                        if the number of rules should not be restricted
        """
        learner = self._create_learner()
        feature_matrix = self._create_row_wise_feature_matrix(x, **kwargs)
        num_outputs = self.num_outputs_

        if learner.can_predict_scores(feature_matrix, num_outputs):
            log.debug('A dense matrix is used to store the predicted scores')
            model = self.model_
            predictor = self._create_score_predictor(learner, model, self.output_space_info_, num_outputs,
                                                     feature_matrix)
            max_rules = int(kwargs.get(self.KWARG_MAX_RULES, 0))

            if predictor.can_predict_incrementally():
                return ClassificationRuleLearner.NativeIncrementalPredictor(
                    feature_matrix, predictor.create_incremental_predictor(max_rules))
            return ClassificationRuleLearner.NonNativeIncrementalPredictor(feature_matrix, model, max_rules, predictor)

        raise RuntimeError('Incremental prediction of scores not supported using the current configuration')

    @staticmethod
    def __create_feature_indices(input_name: str, **kwargs) -> np.ndarray:
        """
        Creates and returns an array that stores features indices.

        :param input_name:  The name of the keyword argument, the indices should be obtained from
        :return:            A `np.ndarray`, shape `(feature_indices)`, that stores the feature indices
        """
        feature_indices = kwargs.get(input_name)

        if feature_indices:
            feature_indices = enforce_dense(feature_indices, order='C', dtype=np.uint32)
            return check_array(feature_indices,
                               ensure_2d=False,
                               dtype=np.uint32,
                               ensure_non_negative=True,
                               input_name=input_name)

        return np.empty(shape=0, dtype=np.uint32)

    def _create_feature_info(self, num_features: int, **kwargs) -> FeatureInfo:
        """
        Creates and returns a `FeatureInfo` that provides information about the types of individual features.

        :param num_features:                The total number of available features
        :keyword nominal_feature_indices:   A `numpy.ndarray`, shape `(num_nominal_features)`, that stores the indices
                                            of all nominal features
        :keyword ordinal_feature_indices:   A `numpy.ndarray`, shape `(num_ordinal_features)`, that stores the indices
                                            of all ordinal features
        :return:                            The `FeatureInfo` that has been created
        """
        ordinal_feature_indices = self.__create_feature_indices(self.KWARG_ORDINAL_FEATURE_INDICES, **kwargs)
        nominal_feature_indices = self.__create_feature_indices(self.KWARG_NOMINAL_FEATURE_INDICES, **kwargs)
        num_ordinal_features = ordinal_feature_indices.shape[0]
        num_nominal_features = nominal_feature_indices.shape[0]

        if num_ordinal_features == 0 and num_nominal_features == 0:
            return EqualFeatureInfo.create_numerical()
        if num_ordinal_features == num_features:
            return EqualFeatureInfo.create_ordinal()
        if num_nominal_features == num_features:
            return EqualFeatureInfo.create_nominal()
        return MixedFeatureInfo(num_features, ordinal_feature_indices, nominal_feature_indices)

    def _create_example_weights(self, num_examples: int, **kwargs) -> ExampleWeights:
        """
        Creates and returns the `ExampleWeights` that provide access to the weights of individual training examples.

        :param num_examples:    The total number of available training examples
        :return:                A `np.ndarray`, shape `(num_examples)`, that provides access to the weights of
                                individual training examples
        """
        example_weights = kwargs.get(self.KWARG_EXAMPLE_WEIGHTS)

        if example_weights:
            example_weights = enforce_dense(example_weights, order='C', dtype=np.float32)
            example_weights = check_array(example_weights,
                                          ensure_2d=False,
                                          dtype=np.float32,
                                          ensure_non_negative=True,
                                          input_name=self.KWARG_EXAMPLE_WEIGHTS)
            return RealValuedExampleWeights(example_weights)

        return EqualExampleWeights(num_examples)

    def _create_column_wise_feature_matrix(self, x, **kwargs) -> ColumnWiseFeatureMatrix:
        """
        Creates and returns a matrix that provides column-wise access to the features of the training examples.

        :param x:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_features)`, that stores the feature values of the training examples
        :return:    The matrix that has been created
        """
        sparse_feature_value = float(kwargs.get(self.KWARG_SPARSE_FEATURE_VALUE, 0.0))
        x_sparse_format = SparseFormat.CSC
        x_sparse_policy = parse_enum('feature_format', self.feature_format, SparsePolicy, default=SparsePolicy.AUTO)
        x_enforce_sparse = x_sparse_policy.should_enforce_sparse(x, sparse_format=x_sparse_format, dtype=np.float32)
        x = x if x_enforce_sparse else enforce_2d(
            enforce_dense(x, order='F', dtype=np.float32, sparse_value=sparse_feature_value))
        x = validate_data(self, X=x, accept_sparse=x_sparse_format, dtype=np.float32, ensure_all_finite='allow-nan')

        if is_sparse(x):
            log.debug(
                'A sparse matrix with sparse value %s is used to store the feature values of the training examples',
                sparse_feature_value)
            x_data = np.ascontiguousarray(x.data, dtype=np.float32)
            x_indices = np.ascontiguousarray(x.indices, dtype=np.uint32)
            x_indptr = np.ascontiguousarray(x.indptr, dtype=np.uint32)
            return CscFeatureMatrix(x_data, x_indices, x_indptr, x.shape[0], x.shape[1], sparse_feature_value)

        log.debug('A dense matrix is used to store the feature values of the training examples')
        return FortranContiguousFeatureMatrix(x)

    def _create_row_wise_feature_matrix(self, x, **kwargs) -> RowWiseFeatureMatrix:
        """
        Creates and returns a matrix that provides row-wise access to the features of query examples.

        :param x:                       A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                        `(num_examples, num_features)`, that stores the feature values of the query
                                        examples
        :keyword sparse_feature_value:  The value that should be used for sparse elements in the feature matrix. Does
                                        only have an effect if `x` is a `scipy.sparse` matrix
        :return:                        A `RowWiseFeatureMatrix` that has been created
        """
        sparse_feature_value = float(kwargs.get(self.KWARG_SPARSE_FEATURE_VALUE, 0.0))
        sparse_format = SparseFormat.CSR
        sparse_policy = parse_enum('feature_format', self.feature_format, SparsePolicy, default=SparsePolicy.AUTO)
        enforce_sparse = sparse_policy.should_enforce_sparse(x, sparse_format=sparse_format, dtype=np.float32)
        x = x if enforce_sparse else enforce_2d(
            enforce_dense(x, order='C', dtype=np.float32, sparse_value=sparse_feature_value))
        x = validate_data(self,
                          X=x,
                          reset=False,
                          accept_sparse=sparse_format,
                          dtype=np.float32,
                          ensure_all_finite='allow-nan')

        if is_sparse(x):
            log.debug('A sparse matrix with sparse value %s is used to store the feature values of the query examples',
                      sparse_feature_value)
            x_data = np.ascontiguousarray(x.data, dtype=np.float32)
            x_indices = np.ascontiguousarray(x.indices, dtype=np.uint32)
            x_indptr = np.ascontiguousarray(x.indptr, dtype=np.uint32)
            return CsrFeatureMatrix(x_data, x_indices, x_indptr, x.shape[0], x.shape[1], sparse_feature_value)

        log.debug('A dense matrix is used to store the feature values of the query examples')
        return CContiguousFeatureMatrix(x)

    def __create_row_wise_output_matrix(self, y) -> Any:
        """
        Must be implemented by subclasses in order to create a matrix that provides row-wise access to the ground truth
        of training examples.

        :param y:   A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                    `(num_examples, num_outputs)`, that stores the ground truth of the training examples
        :return:    The matrix that has been created
        """
        y_sparse_format = SparseFormat.CSR
        prediction_sparse_policy = parse_enum('prediction_format',
                                              self.prediction_format,
                                              SparsePolicy,
                                              default=SparsePolicy.AUTO)
        self.sparse_predictions_ = prediction_sparse_policy != SparsePolicy.FORCE_DENSE and (
            prediction_sparse_policy == SparsePolicy.FORCE_SPARSE
            or is_sparse_and_memory_efficient(y, sparse_format=y_sparse_format, dtype=np.uint8, sparse_values=False))

        y_sparse_policy = parse_enum('output_format', self.output_format, SparsePolicy, default=SparsePolicy.AUTO)
        y_enforce_sparse = y_sparse_policy.should_enforce_sparse(y,
                                                                 sparse_format=y_sparse_format,
                                                                 dtype=np.uint8,
                                                                 sparse_values=False)
        return self._create_row_wise_output_matrix(y, sparse_format=y_sparse_format, sparse=y_enforce_sparse)

    @abstractmethod
    def _create_row_wise_output_matrix(self, y, sparse_format: SparseFormat, sparse: bool, **kwargs) -> Any:
        """
        Must be implemented by subclasses in order to create a matrix that provides row-wise access to the ground truth
        of training examples.

        :param y:               A `numpy.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray`, shape
                                `(num_examples, num_outputs)`, that stores the ground truth of the training examples
        :param sparse_format:   The `SparseFormat` to be used for sparse matrices
        :param sparse:          True, if the given matrix should be converted into a sparse matrix, False otherwise
        :return:                The matrix that has been created
        """

    @abstractmethod
    def _create_learner(self) -> Any:
        """
        Must be implemented by subclasses in order to configure and create an implementation of the rule learner.

        :return: The implementation of the rule learner that has been created
        """


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


class ClassificationRuleLearner(RuleLearner, ClassifierMixin, IncrementalClassifierMixin, ABC):
    """
    A scikit-learn implementation of a rule learning algorithm that can be applied to classification problems.
    """

    class NativeIncrementalProbabilityPredictor(RuleLearner.NativeIncrementalPredictor):
        """
        Allows to obtain probability estimates from a `ClassificationRuleLearner` incrementally by using its native
        support of this functionality.
        """

        def apply_next(self, step_size: int):
            return convert_into_sklearn_compatible_probabilities(super().apply_next(step_size))

    class NonNativeIncrementalProbabilityPredictor(RuleLearner.NonNativeIncrementalPredictor):
        """
        Allows to obtain probability estimates from a `ClassificationRuleLearner` incrementally.
        """

        def apply_next(self, step_size: int):
            return convert_into_sklearn_compatible_probabilities(super().apply_next(step_size))

    def _create_row_wise_output_matrix(self, y, sparse_format: SparseFormat, sparse: bool, **_) -> Any:
        y = check_array(y if sparse else enforce_2d(enforce_dense(y, order='C', dtype=np.uint8)),
                        accept_sparse=sparse_format,
                        dtype=np.uint8,
                        ensure_non_negative=True)

        if is_sparse(y):
            log.debug('A sparse matrix is used to store the labels of the training examples')
            y_indices = np.ascontiguousarray(y.indices, dtype=np.uint32)
            y_indptr = np.ascontiguousarray(y.indptr, dtype=np.uint32)
            return CsrLabelMatrix(y_indices, y_indptr, y.shape[0], y.shape[1])

        log.debug('A dense matrix is used to store the labels of the training examples')
        return CContiguousLabelMatrix(y)

    @staticmethod
    def _create_probability_predictor(learner: RuleLearnerWrapper, model: RuleModel, output_space_info: OutputSpaceInfo,
                                      marginal_probability_calibration_model: MarginalProbabilityCalibrationModel,
                                      joint_probability_calibration_model: JointProbabilityCalibrationModel,
                                      num_labels: int, feature_matrix: RowWiseFeatureMatrix):
        """
        Creates and returns a predictor for predicting probability estimates.

        :param learner:                                 The learner for which the predictor should be created
        :param model:                                   The model to be used for prediction
        :param output_space_info:                       Information about the output space that may be used for
                                                        prediction
        :param marginal_probability_calibration_model:  A model for the calibration of marginal probabilities
        :param joint_probability_calibration_model:     A model for the calibration of joint probabilities
        :param num_labels:                              The total number of labels to predict for
        :param feature_matrix:                          A feature matrix that provides row-wise access to the features
                                                        of the query examples
        :return:                                        The predictor that has been created
        """
        return learner.create_probability_predictor(feature_matrix, model, output_space_info,
                                                    marginal_probability_calibration_model,
                                                    joint_probability_calibration_model, num_labels)

    def _predict_proba(self, x, **kwargs):
        """
        :keyword sparse_feature_value: The value that should be used for sparse elements in the feature matrix. Does
                                       only have an effect if `x` is a `scipy.sparse.spmatrix` or `scipy.sparse.sparray`
        """
        learner = self._create_learner()
        feature_matrix = self._create_row_wise_feature_matrix(x, **kwargs)
        num_outputs = self.num_outputs_

        if learner.can_predict_probabilities(feature_matrix, num_outputs):
            log.debug('A dense matrix is used to store the predicted probability estimates')
            max_rules = int(kwargs.get(self.KWARG_MAX_RULES, 0))
            return convert_into_sklearn_compatible_probabilities(
                self._create_probability_predictor(learner, self.model_, self.output_space_info_,
                                                   self.marginal_probability_calibration_model_,
                                                   self.joint_probability_calibration_model_, num_outputs,
                                                   feature_matrix).predict(max_rules))

        return super()._predict_proba(x, **kwargs)

    def _predict_proba_incrementally(self, x, **kwargs):
        """
        :keyword sparse_feature_value:  The value that should be used for sparse elements in the feature matrix. Does
                                        only have an effect if `x` is a `scipy.sparse.spmatrix` or
                                        `scipy.sparse.sparray`
        :keyword max_rules:             The maximum number of rules to be used for prediction. Must be at least 1 or 0,
                                        if the number of rules should not be restricted
        """
        learner = self._create_learner()
        feature_matrix = self._create_row_wise_feature_matrix(x, **kwargs)
        num_outputs = self.num_outputs_

        if learner.can_predict_probabilities(feature_matrix, num_outputs):
            log.debug('A dense matrix is used to store the predicted probability estimates')
            model = self.model_
            predictor = self._create_probability_predictor(learner, model, self.output_space_info_,
                                                           self.marginal_probability_calibration_model_,
                                                           self.joint_probability_calibration_model_, num_outputs,
                                                           feature_matrix)
            max_rules = int(kwargs.get(self.KWARG_MAX_RULES, 0))

            if predictor.can_predict_incrementally():
                return ClassificationRuleLearner.NativeIncrementalProbabilityPredictor(
                    feature_matrix, predictor.create_incremental_predictor(max_rules))
            return ClassificationRuleLearner.NonNativeIncrementalProbabilityPredictor(
                feature_matrix, model, max_rules, predictor)

        return super().predict_proba_incrementally(x, **kwargs)

    @staticmethod
    def _create_binary_predictor(learner: RuleLearnerWrapper, model: RuleModel, output_space_info: OutputSpaceInfo,
                                 marginal_probability_calibration_model: MarginalProbabilityCalibrationModel,
                                 joint_probability_calibration_model: JointProbabilityCalibrationModel, num_labels: int,
                                 feature_matrix: RowWiseFeatureMatrix, sparse: bool):
        """
        Creates and returns a predictor for predicting binary labels.

        :param learner:                                 The learner for which the predictor should be created
        :param model:                                   The model to be used for prediction
        :param output_space_info:                       Information about the output space that may be used for
                                                        prediction
        :param marginal_probability_calibration_model:  A model for the calibration of marginal probabilities
        :param joint_probability_calibration_model:     A model for the calibration of joint probabilities
        :param num_labels:                              The total number of labels to predict for
        :param feature_matrix:                          A feature matrix that provides row-wise access to the features
                                                        of the query examples
        :param sparse:                                  True, if a sparse matrix should be used for storing predictions,
                                                        False otherwise
        :return:                                        The predictor that has been created
        """
        if sparse:
            return learner.create_sparse_binary_predictor(feature_matrix, model, output_space_info,
                                                          marginal_probability_calibration_model,
                                                          joint_probability_calibration_model, num_labels)
        return learner.create_binary_predictor(feature_matrix, model, output_space_info,
                                               marginal_probability_calibration_model,
                                               joint_probability_calibration_model, num_labels)

    def _predict_binary(self, x, **kwargs):
        """
        :keyword sparse_feature_value: The value that should be used for sparse elements in the feature matrix. Does
                                       only have an effect if `x` is a `scipy.sparse.spmatrix` or `scipy.sparse.sparray`
        """
        learner = self._create_learner()
        feature_matrix = self._create_row_wise_feature_matrix(x, **kwargs)
        num_outputs = self.num_outputs_

        if learner.can_predict_binary(feature_matrix, num_outputs):
            sparse_predictions = self.sparse_predictions_
            log.debug('A %s matrix is used to store the predicted labels', 'sparse' if sparse_predictions else 'dense')
            max_rules = int(kwargs.get(self.KWARG_MAX_RULES, 0))
            return self._create_binary_predictor(learner, self.model_, self.output_space_info_,
                                                 self.marginal_probability_calibration_model_,
                                                 self.joint_probability_calibration_model_, num_outputs, feature_matrix,
                                                 sparse_predictions).predict(max_rules)

        return super()._predict_binary(x, **kwargs)

    def _predict_binary_incrementally(self, x, **kwargs):
        """
        :keyword sparse_feature_value:  The value that should be used for sparse elements in the feature matrix. Does
                                        only have an effect if `x` is a `scipy.sparse.spmatrix` or
                                        `scipy.sparse.sparray`
        :keyword max_rules:             The maximum number of rules to be used for prediction. Must be at least 1 or 0,
                                        if the number of rules should not be restricted
        """
        learner = self._create_learner()
        feature_matrix = self._create_row_wise_feature_matrix(x, **kwargs)
        num_outputs = self.num_outputs_

        if learner.can_predict_binary(feature_matrix, num_outputs):
            sparse_predictions = self.sparse_predictions_
            log.debug('A %s matrix is used to store the predicted labels', 'sparse' if sparse_predictions else 'dense')
            model = self.model_
            predictor = self._create_binary_predictor(learner, model, self.output_space_info_,
                                                      self.marginal_probability_calibration_model_,
                                                      self.joint_probability_calibration_model_, num_outputs,
                                                      feature_matrix, sparse_predictions)
            max_rules = int(kwargs.get(self.KWARG_MAX_RULES, 0))

            if predictor.can_predict_incrementally():
                return ClassificationRuleLearner.NativeIncrementalPredictor(
                    feature_matrix, predictor.create_incremental_predictor(max_rules))
            return ClassificationRuleLearner.NonNativeIncrementalPredictor(feature_matrix, model, max_rules, predictor)

        return super()._predict_binary_incrementally(x, **kwargs)


class RegressionRuleLearner(RuleLearner, RegressorMixin, IncrementalRegressorMixin, ABC):
    """
    A scikit-learn implementation of a rule learning algorithm that can be applied to regression problems.
    """

    def _create_row_wise_output_matrix(self, y, sparse_format: SparseFormat, sparse: bool, **_) -> Any:
        y = check_array(y if sparse else enforce_2d(enforce_dense(y, order='C', dtype=np.float32)),
                        accept_sparse=sparse_format,
                        dtype=np.float32)

        if is_sparse(y):
            log.debug('A sparse matrix is used to store the regression scores of the training examples')
            y_data = np.ascontiguousarray(y.data, dtype=np.float32)
            y_indices = np.ascontiguousarray(y.indices, dtype=np.uint32)
            y_indptr = np.ascontiguousarray(y.indptr, dtype=np.uint32)
            return CsrRegressionMatrix(y_data, y_indices, y_indptr, y.shape[0], y.shape[1])

        log.debug('A dense matrix is used to store the regression scores of the training examples')
        return CContiguousRegressionMatrix(y)


def configure_rule_learner(learner: RuleLearner, config: RuleLearnerConfig, parameters: Set[Parameter]):
    """
    Configures a rule learner by taking into account a given set of parameters.

    :param learner:     The rule learner to be configured
    :param config:      The configuration to be modified
    :param parameters:  A set that contains the parameters to be taken into account
    """
    for parameter in parameters:
        parameter_name = parameter.name

        if hasattr(learner, parameter_name):
            value = getattr(learner, parameter_name)

            if value is not None:
                parameter.configure(config=config, value=value)
