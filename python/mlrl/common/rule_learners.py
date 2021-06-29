#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for implementing single- or multi-label rule learning algorithms.
"""
import logging as log
import os
from abc import abstractmethod
from ast import literal_eval
from enum import Enum
from typing import List

import numpy as np
from mlrl.common.cython.binning import EqualWidthFeatureBinning, EqualFrequencyFeatureBinning
from mlrl.common.cython.input import BitNominalFeatureMask, EqualNominalFeatureMask
from mlrl.common.cython.input import FortranContiguousFeatureMatrix, CscFeatureMatrix, CsrFeatureMatrix, \
    CContiguousFeatureMatrix
from mlrl.common.cython.input import LabelMatrix, CContiguousLabelMatrix, CsrLabelMatrix
from mlrl.common.cython.input import LabelVectorSet
from mlrl.common.cython.model import ModelBuilder
from mlrl.common.cython.output import Predictor
from mlrl.common.cython.pruning import Pruning, NoPruning, IREP
from mlrl.common.cython.rule_induction import RuleModelInduction
from mlrl.common.cython.sampling import FeatureSamplingFactory, FeatureSamplingWithoutReplacementFactory, \
    NoFeatureSamplingFactory
from mlrl.common.cython.sampling import InstanceSamplingFactory, InstanceSamplingWithReplacementFactory, \
    InstanceSamplingWithoutReplacementFactory, NoInstanceSamplingFactory, LabelWiseStratifiedSamplingFactory, \
    ExampleWiseStratifiedSamplingFactory
from mlrl.common.cython.sampling import LabelSamplingFactory, LabelSamplingWithoutReplacementFactory, \
    NoLabelSamplingFactory
from mlrl.common.cython.sampling import PartitionSamplingFactory, NoPartitionSamplingFactory, \
    RandomBiPartitionSamplingFactory, LabelWiseStratifiedBiPartitionSamplingFactory, \
    ExampleWiseStratifiedBiPartitionSamplingFactory
from mlrl.common.cython.stopping import StoppingCriterion, SizeStoppingCriterion, TimeStoppingCriterion
from mlrl.common.cython.thresholds import ThresholdsFactory
from mlrl.common.cython.thresholds_approximate import ApproximateThresholdsFactory
from mlrl.common.cython.thresholds_exact import ExactThresholdsFactory
from scipy.sparse import issparse, isspmatrix_lil, isspmatrix_coo, isspmatrix_dok, isspmatrix_csc, isspmatrix_csr
from sklearn.utils import check_array

from mlrl.common.arrays import enforce_dense
from mlrl.common.learners import Learner, NominalAttributeLearner
from mlrl.common.types import DTYPE_UINT8, DTYPE_UINT32, DTYPE_FLOAT32

HEAD_REFINEMENT_SINGLE = 'single-label'

SAMPLING_WITH_REPLACEMENT = 'with-replacement'

SAMPLING_WITHOUT_REPLACEMENT = 'without-replacement'

SAMPLING_STRATIFIED_LABEL_WISE = 'stratified-label-wise'

SAMPLING_STRATIFIED_EXAMPLE_WISE = 'stratified-example-wise'

ARGUMENT_SAMPLE_SIZE = 'sample_size'

ARGUMENT_NUM_SAMPLES = 'num_samples'

PARTITION_SAMPLING_RANDOM = 'random'

ARGUMENT_HOLDOUT_SET_SIZE = 'holdout_set_size'

BINNING_EQUAL_FREQUENCY = 'equal-frequency'

BINNING_EQUAL_WIDTH = 'equal-width'

ARGUMENT_BIN_RATIO = 'bin_ratio'

ARGUMENT_MIN_BINS = 'min_bins'

ARGUMENT_MAX_BINS = 'max_bins'

PRUNING_IREP = 'irep'


class SparsePolicy(Enum):
    AUTO = 'auto'
    FORCE_SPARSE = 'sparse'
    FORCE_DENSE = 'dense'


class SparseFormat(Enum):
    CSC = 'csc'
    CSR = 'csr'


def create_sparse_policy(policy: str) -> SparsePolicy:
    try:
        return SparsePolicy(policy)
    except ValueError:
        raise ValueError('Invalid matrix format given: \'' + str(policy) + '\'. Must be one of ' + str(
            [x.value for x in SparsePolicy]))


def create_label_sampling_factory(label_sampling: str, num_labels: int) -> LabelSamplingFactory:
    if label_sampling is None:
        return NoLabelSamplingFactory()
    else:
        prefix, args = parse_prefix_and_dict(label_sampling, [SAMPLING_WITHOUT_REPLACEMENT])

        if prefix == SAMPLING_WITHOUT_REPLACEMENT:
            num_samples = get_int_argument(args, ARGUMENT_NUM_SAMPLES, 1, lambda x: 1 <= x < num_labels)
            return LabelSamplingWithoutReplacementFactory(num_samples)
        raise ValueError('Invalid value given for parameter \'label_sampling\': ' + str(label_sampling))


def create_instance_sampling_factory(instance_sampling: str) -> InstanceSamplingFactory:
    if instance_sampling is None:
        return NoInstanceSamplingFactory()
    else:
        prefix, args = parse_prefix_and_dict(instance_sampling,
                                             [SAMPLING_WITH_REPLACEMENT, SAMPLING_WITHOUT_REPLACEMENT,
                                              SAMPLING_STRATIFIED_LABEL_WISE, SAMPLING_STRATIFIED_EXAMPLE_WISE])

        if prefix == SAMPLING_WITH_REPLACEMENT:
            sample_size = get_float_argument(args, ARGUMENT_SAMPLE_SIZE, 1.0, lambda x: 0 < x <= 1)
            return InstanceSamplingWithReplacementFactory(sample_size)
        elif prefix == SAMPLING_WITHOUT_REPLACEMENT:
            sample_size = get_float_argument(args, ARGUMENT_SAMPLE_SIZE, 0.66, lambda x: 0 < x < 1)
            return InstanceSamplingWithoutReplacementFactory(sample_size)
        elif prefix == SAMPLING_STRATIFIED_LABEL_WISE:
            sample_size = get_float_argument(args, ARGUMENT_SAMPLE_SIZE, 0.66, lambda x: 0 < x < 1)
            return LabelWiseStratifiedSamplingFactory(sample_size)
        elif prefix == SAMPLING_STRATIFIED_EXAMPLE_WISE:
            sample_size = get_float_argument(args, ARGUMENT_SAMPLE_SIZE, 0.66, lambda x: 0 < x < 1)
            return ExampleWiseStratifiedSamplingFactory(sample_size)
        raise ValueError('Invalid value given for parameter \'instance_sampling\': ' + str(instance_sampling))


def create_feature_sampling_factory(feature_sampling: str) -> FeatureSamplingFactory:
    if feature_sampling is None:
        return NoFeatureSamplingFactory()
    else:
        prefix, args = parse_prefix_and_dict(feature_sampling, [SAMPLING_WITHOUT_REPLACEMENT])

        if prefix == SAMPLING_WITHOUT_REPLACEMENT:
            sample_size = get_float_argument(args, ARGUMENT_SAMPLE_SIZE, 0.0, lambda x: 0 <= x < 1)
            return FeatureSamplingWithoutReplacementFactory(sample_size)
        raise ValueError('Invalid value given for parameter \'feature_sampling\': ' + str(feature_sampling))


def create_partition_sampling_factory(holdout: str) -> PartitionSamplingFactory:
    if holdout is None:
        return NoPartitionSamplingFactory()
    else:
        prefix, args = parse_prefix_and_dict(holdout, [PARTITION_SAMPLING_RANDOM, SAMPLING_STRATIFIED_LABEL_WISE,
                                                       SAMPLING_STRATIFIED_EXAMPLE_WISE])

        if prefix == PARTITION_SAMPLING_RANDOM:
            holdout_set_size = get_float_argument(args, ARGUMENT_HOLDOUT_SET_SIZE, 0.33, lambda x: 0 < x < 1)
            return RandomBiPartitionSamplingFactory(holdout_set_size)
        if prefix == SAMPLING_STRATIFIED_LABEL_WISE:
            holdout_set_size = get_float_argument(args, ARGUMENT_HOLDOUT_SET_SIZE, 0.33, lambda x: 0 < x < 1)
            return LabelWiseStratifiedBiPartitionSamplingFactory(holdout_set_size)
        if prefix == SAMPLING_STRATIFIED_EXAMPLE_WISE:
            holdout_set_size = get_float_argument(args, ARGUMENT_HOLDOUT_SET_SIZE, 0.33, lambda x: 0 < x < 1)
            return ExampleWiseStratifiedBiPartitionSamplingFactory(holdout_set_size)
        raise ValueError('Invalid value given for parameter \'holdout\': ' + str(holdout))


def create_pruning(pruning: str, instance_sampling: str) -> Pruning:
    if pruning is None:
        return NoPruning()
    else:
        if pruning == PRUNING_IREP:
            if instance_sampling is None:
                log.warning('Parameter \'pruning\' does not have any effect, because parameter '
                            + '\'instance_sampling\' is set to \'None\'!')
                return NoPruning()
            return IREP()
        raise ValueError('Invalid value given for parameter \'pruning\': ' + str(pruning))


def create_stopping_criteria(max_rules: int, time_limit: int) -> List[StoppingCriterion]:
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


def create_min_coverage(min_coverage: int) -> int:
    if min_coverage < 1:
        raise ValueError('Invalid value given for parameter \'min_coverage\': ' + str(min_coverage))

    return min_coverage


def create_max_conditions(max_conditions: int) -> int:
    if max_conditions != -1 and max_conditions < 1:
        raise ValueError('Invalid value given for parameter \'max_conditions\': ' + str(max_conditions))

    return max_conditions


def create_max_head_refinements(max_head_refinements: int) -> int:
    if max_head_refinements != -1 and max_head_refinements < 1:
        raise ValueError('Invalid value given for parameter \'max_head_refinements\': ' + str(max_head_refinements))

    return max_head_refinements


def get_preferred_num_threads(num_threads: int) -> int:
    if num_threads == -1:
        return os.cpu_count()
    if num_threads < 1:
        raise ValueError('Invalid number of threads given: ' + str(num_threads))

    return num_threads


def create_thresholds_factory(feature_binning: str, num_threads: int) -> ThresholdsFactory:
    if feature_binning is None:
        return ExactThresholdsFactory(num_threads)
    else:
        prefix, args = parse_prefix_and_dict(feature_binning, [BINNING_EQUAL_FREQUENCY, BINNING_EQUAL_WIDTH])

        if prefix == BINNING_EQUAL_FREQUENCY:
            bin_ratio = get_float_argument(args, ARGUMENT_BIN_RATIO, 0.33, lambda x: 0 < x < 1)
            min_bins = get_int_argument(args, ARGUMENT_MIN_BINS, 2, lambda x: x >= 2)
            max_bins = get_int_argument(args, ARGUMENT_MAX_BINS, 0, lambda x: x == 0 or x >= min_bins)
            return ApproximateThresholdsFactory(EqualFrequencyFeatureBinning(bin_ratio, min_bins, max_bins),
                                                num_threads)
        elif prefix == BINNING_EQUAL_WIDTH:
            bin_ratio = get_float_argument(args, ARGUMENT_BIN_RATIO, 0.33, lambda x: 0 < x < 1)
            min_bins = get_int_argument(args, ARGUMENT_MIN_BINS, 2, lambda x: x >= 2)
            max_bins = get_int_argument(args, ARGUMENT_MAX_BINS, 0, lambda x: x == 0 or x >= min_bins)
            return ApproximateThresholdsFactory(EqualWidthFeatureBinning(bin_ratio, min_bins, max_bins), num_threads)
        raise ValueError('Invalid value given for parameter \'feature_binning\': ' + str(feature_binning))


def parse_prefix_and_dict(string: str, prefixes: List[str]) -> [str, dict]:
    for prefix in prefixes:
        if string.startswith(prefix):
            suffix = string[len(prefix):].strip()

            if len(suffix) > 0:
                return prefix, literal_eval(suffix)

            return prefix, {}

    return None, None


def get_string_argument(args: dict, key: str, default: str, validation=None) -> str:
    if args is not None and key in args:
        value = str(args[key])

        if validation is not None and not validation(value):
            raise ValueError('Invalid value given for string argument \'' + key + '\': ' + str(value))

        return value

    return default


def get_bool_argument(args: dict, key: str, default: bool) -> bool:
    if args is not None and key in args:
        return bool(args[key])

    return default


def get_int_argument(args: dict, key: str, default: int, validation=None) -> int:
    if args is not None and key in args:
        value = int(args[key])

        if validation is not None and not validation(value):
            raise ValueError('Invalid value given for int argument \'' + key + '\': ' + str(value))

        return value

    return default


def get_float_argument(args: dict, key: str, default: float, validation=None) -> float:
    if args is not None and key in args:
        value = float(args[key])

        if validation is not None and not validation(value):
            raise ValueError('Invalid value given for float argument \'' + key + '\': ' + str(value))

        return value

    return default


def should_enforce_sparse(m, sparse_format: SparseFormat, policy: SparsePolicy, dtype,
                          sparse_values: bool = True) -> bool:
    """
    Returns whether it is preferable to convert a given matrix into a `scipy.sparse.csr_matrix`,
    `scipy.sparse.csc_matrix` or `scipy.sparse.dok_matrix`, depending on the format of the given matrix and a given
    `SparsePolicy`:

    - If the given policy is `SparsePolicy.AUTO`, the matrix will be converted into the given sparse format, if
      possible, if the sparse matrix is expected to occupy less memory than a dense matrix. To be able to convert the
      matrix into a sparse format, it must be a `scipy.sparse.lil_matrix`, `scipy.sparse.dok_matrix` or
      `scipy.sparse.coo_matrix`. If the given sparse format is `csr` or `csc` and the matrix is a already in that
      format, it will not be converted.

    - If the given policy is `SparsePolicy.FORCE_DENSE`, the matrix will always be converted into the specified sparse
    format, if possible.

    - If the given policy is `SparsePolicy.FORCE_SPARSE`, the matrix will always be converted into a dense matrix.

    :param m:               A `np.ndarray` or `scipy.sparse.matrix` to be checked
    :param sparse_format:   The `SparseFormat` to be used
    :param policy:          The `SparsePolicy` to be used
    :param dtype            The type of the values that should be stored in the matrix
    :param sparse_values:   True, if the values must explicitly be stored when using a sparse format, False otherwise
    :return:                True, if it is preferable to convert the matrix into a sparse matrix of the given format,
                            False otherwise
    """
    if not issparse(m):
        # Given matrix is dense
        if policy != SparsePolicy.FORCE_SPARSE:
            return False
    elif (isspmatrix_csr(m) and sparse_format == SparseFormat.CSR) or (
            isspmatrix_csc(m) and sparse_format == SparseFormat.CSC):
        # Matrix is a `scipy.sparse.csr_matrix` or `scipy.sparse.csc_matrix` and is already in the given sparse format
        return policy != SparsePolicy.FORCE_DENSE
    elif isspmatrix_lil(m) or isspmatrix_coo(m) or isspmatrix_dok(m):
        # Given matrix is in a format that might be converted into the specified sparse format
        if policy == SparsePolicy.AUTO:
            num_pointers = m.shape[1 if sparse_format == SparseFormat.CSC else 0]
            size_int = np.dtype(DTYPE_UINT32).itemsize
            size_data = np.dtype(dtype).itemsize if sparse_values else 0
            num_non_zero = m.nnz
            size_sparse = (num_non_zero * size_data) + (num_non_zero * size_int) + (num_pointers * size_int)
            size_dense = np.prod(m.shape) * size_data
            return size_sparse < size_dense
        else:
            return policy == SparsePolicy.FORCE_SPARSE

    raise ValueError(
        'Matrix of type ' + type(m).__name__ + ' cannot be converted to format \'' + str(sparse_format) + '\'')


class MLRuleLearner(Learner, NominalAttributeLearner):
    """
    A scikit-multilearn implementation of a rule learning algorithm for multi-label classification or ranking.

    Attributes
        label_vectors_          A `LabelVectorSet` that stores the known label vectors, if necessary
        predictor_              The `Predictor` to be used for making predictions
        probability_predictor_  The `Predictor` to be used for predicting probability estimates
    """

    def __init__(self, random_state: int, feature_format: str, label_format: str):
        """
        :param random_state:    The seed to be used by RNGs. Must be at least 1
        :param feature_format:  The format to be used for the feature matrix. Must be 'sparse', 'dense' or 'auto'
        :param label_format:    The format to be used for the label matrix. Must be 'sparse', 'dense' or 'auto'
        """
        super().__init__()
        self.random_state = random_state
        self.feature_format = feature_format
        self.label_format = label_format

    def _fit(self, x, y):
        # Validate feature matrix and convert it to the preferred format...
        x_sparse_format = SparseFormat.CSC
        x_sparse_policy = create_sparse_policy(self.feature_format)
        x_enforce_sparse = should_enforce_sparse(x, sparse_format=x_sparse_format, policy=x_sparse_policy,
                                                 dtype=DTYPE_FLOAT32)
        x = self._validate_data((x if x_enforce_sparse else enforce_dense(x, order='F', dtype=DTYPE_FLOAT32)),
                                accept_sparse=(x_sparse_format.value if x_enforce_sparse else False),
                                dtype=DTYPE_FLOAT32, force_all_finite='allow-nan')
        num_features = x.shape[1]

        if issparse(x):
            log.debug('A sparse matrix is used to store the feature values of the training examples')
            x_data = np.ascontiguousarray(x.data, dtype=DTYPE_FLOAT32)
            x_row_indices = np.ascontiguousarray(x.indices, dtype=DTYPE_UINT32)
            x_col_indices = np.ascontiguousarray(x.indptr, dtype=DTYPE_UINT32)
            feature_matrix = CscFeatureMatrix(x.shape[0], x.shape[1], x_data, x_row_indices, x_col_indices)
        else:
            log.debug('A dense matrix is used to store the feature values of the training examples')
            feature_matrix = FortranContiguousFeatureMatrix(x)

        # Validate label matrix and convert it to the preferred format...
        y_sparse_format = SparseFormat.CSR
        y_sparse_policy = create_sparse_policy(self.label_format)
        y_enforce_sparse = should_enforce_sparse(y, sparse_format=y_sparse_format, policy=y_sparse_policy,
                                                 dtype=DTYPE_UINT8, sparse_values=False)
        y = check_array((y if y_enforce_sparse else y.toarray(order='C')),
                        accept_sparse=(y_sparse_format.value if y_enforce_sparse else False), ensure_2d=False,
                        dtype=DTYPE_UINT8)
        num_labels = y.shape[1]

        if issparse(y):
            log.debug('A sparse matrix is used to store the labels of the training examples')
            y_row_indices = np.ascontiguousarray(y.indptr, dtype=DTYPE_UINT32)
            y_col_indices = np.ascontiguousarray(y.indices, dtype=DTYPE_UINT32)
            label_matrix = CsrLabelMatrix(y.shape[0], y.shape[1], y_row_indices, y_col_indices)
        else:
            log.debug('A dense matrix is used to store the labels of the training examples')
            label_matrix = CContiguousLabelMatrix(y)

        # Create predictors...
        self.predictor_ = self._create_predictor(num_labels)
        self.probability_predictor_ = self._create_probability_predictor(num_labels)
        self.label_vectors_ = self._create_label_vector_set(label_matrix)

        # Create a mask that provides access to the information whether individual features are nominal or not...
        if self.nominal_attribute_indices is None or len(self.nominal_attribute_indices) == 0:
            nominal_feature_mask = EqualNominalFeatureMask(False)
        elif len(self.nominal_attribute_indices) == num_features:
            nominal_feature_mask = EqualNominalFeatureMask(True)
        else:
            nominal_feature_mask = BitNominalFeatureMask(num_features, self.nominal_attribute_indices)

        # Induce rules...
        rule_model_induction = self._create_rule_model_induction(num_labels)
        model_builder = self._create_model_builder()
        return rule_model_induction.induce_rules(nominal_feature_mask, feature_matrix, label_matrix, self.random_state,
                                                 model_builder)

    def _predict(self, x):
        predictor = self.predictor_
        label_vectors = self.label_vectors_
        return self.__predict(predictor, label_vectors, x)

    def _predict_proba(self, x):
        predictor = self.probability_predictor_

        if predictor is None:
            return super()._predict_proba(x)
        else:
            label_vectors = self.label_vectors_
            return self.__predict(predictor, label_vectors, x)

    def __predict(self, predictor, label_vectors: LabelVectorSet, x):
        sparse_format = SparseFormat.CSR
        sparse_policy = create_sparse_policy(self.feature_format)
        enforce_sparse = should_enforce_sparse(x, sparse_format=sparse_format, policy=sparse_policy,
                                               dtype=DTYPE_FLOAT32)
        x = self._validate_data(x if enforce_sparse else enforce_dense(x, order='C', dtype=DTYPE_FLOAT32), reset=False,
                                accept_sparse=(sparse_format.value if enforce_sparse else False), dtype=DTYPE_FLOAT32,
                                force_all_finite='allow-nan')
        model = self.model_

        if issparse(x):
            log.debug('A sparse matrix is used to store the feature values of the test examples')
            x_data = np.ascontiguousarray(x.data, dtype=DTYPE_FLOAT32)
            x_row_indices = np.ascontiguousarray(x.indptr, dtype=DTYPE_UINT32)
            x_col_indices = np.ascontiguousarray(x.indices, dtype=DTYPE_UINT32)
            feature_matrix = CsrFeatureMatrix(x.shape[0], x.shape[1], x_data, x_row_indices, x_col_indices)
            return predictor.predict_csr(feature_matrix, model, label_vectors)
        else:
            log.debug('A dense matrix is used to store the feature values of the test examples')
            feature_matrix = CContiguousFeatureMatrix(x)
            return predictor.predict(feature_matrix, model, label_vectors)

    @abstractmethod
    def _create_predictor(self, num_labels: int) -> Predictor:
        """
        Must be implemented by subclasses in order to create the `Predictor` to be used for making predictions.

        :param num_labels:  The number of labels in the training data set
        :return:            The `Predictor` that has been created
        """
        pass

    def _create_probability_predictor(self, num_labels: int) -> Predictor:
        """
        Must be implemented by subclasses in order to create the `Predictor` to be used for predicting probability
        estimates.

        :param num_labels:  The number of labels in the training data set
        :return:            The `Predictor` that has been created or None, if the prediction of probabilities is not
                            supported
        """
        return None

    def _create_label_vector_set(self, label_matrix: LabelMatrix) -> LabelVectorSet:
        """
        Must be implemented by subclasses in order to create a `LabelVectorSet` that stores all known label vectors.

        :param label_matrix:    The label matrix that provides access to the labels of the training examples
        :return:                The `LabelVectorSet` that has been created or None, if no such set should be used
        """
        return None

    @abstractmethod
    def _create_rule_model_induction(self, num_labels: int) -> RuleModelInduction:
        """
        Must be implemented by subclasses in order to create the algorithm that should be used for inducing a rule
        model.

        :param num_labels:  The number of labels in the training data set
        :return:            The algorithm for inducting a rule model that has been created
        """
        pass

    @abstractmethod
    def _create_model_builder(self) -> ModelBuilder:
        """
        Must be implemented by subclasses in order to create the builder that should be used for building the model.

        :return: The builder that has been created
        """
        pass
