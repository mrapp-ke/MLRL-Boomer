#!/usr/bin/python

"""
Author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for implementing single- or multi-label rule learning algorithms.
"""
import logging as log
import os
from abc import abstractmethod
from enum import Enum
from typing import List, Dict, Set

import numpy as np
from mlrl.common.cython.binning import EqualWidthFeatureBinning, EqualFrequencyFeatureBinning
from mlrl.common.cython.feature_sampling import FeatureSamplingFactory, FeatureSamplingWithoutReplacementFactory, \
    NoFeatureSamplingFactory
from mlrl.common.cython.input import BitNominalFeatureMask, EqualNominalFeatureMask
from mlrl.common.cython.input import FortranContiguousFeatureMatrix, CscFeatureMatrix, CsrFeatureMatrix, \
    CContiguousFeatureMatrix
from mlrl.common.cython.input import LabelMatrix, CContiguousLabelMatrix, CsrLabelMatrix
from mlrl.common.cython.input import LabelVectorSet
from mlrl.common.cython.model import ModelBuilder
from mlrl.common.cython.output import Predictor
from mlrl.common.cython.partition_sampling import PartitionSamplingFactory, NoPartitionSamplingFactory, \
    RandomBiPartitionSamplingFactory, LabelWiseStratifiedBiPartitionSamplingFactory, \
    ExampleWiseStratifiedBiPartitionSamplingFactory
from mlrl.common.cython.pruning import Pruning, NoPruning, IREP
from mlrl.common.cython.rule_model_assemblage import RuleModelAssemblage
from mlrl.common.cython.sampling import LabelSamplingFactory, LabelSamplingWithoutReplacementFactory, \
    NoLabelSamplingFactory
from mlrl.common.cython.stopping import StoppingCriterion, SizeStoppingCriterion, TimeStoppingCriterion
from mlrl.common.cython.thresholds import ThresholdsFactory
from mlrl.common.cython.thresholds_approximate import ApproximateThresholdsFactory
from mlrl.common.cython.thresholds_exact import ExactThresholdsFactory
from scipy.sparse import issparse, isspmatrix_lil, isspmatrix_coo, isspmatrix_dok, isspmatrix_csc, isspmatrix_csr
from sklearn.utils import check_array

from mlrl.common.arrays import enforce_dense
from mlrl.common.learners import Learner, NominalAttributeLearner
from mlrl.common.options import Options
from mlrl.common.strings import format_enum_values, format_string_set, format_dict_keys
from mlrl.common.types import DTYPE_UINT8, DTYPE_UINT32, DTYPE_FLOAT32

AUTOMATIC = 'auto'

HEAD_TYPE_SINGLE = 'single-label'

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

LABEL_SAMPLING_VALUES: Dict[str, Set[str]] = {
    SAMPLING_WITHOUT_REPLACEMENT: {ARGUMENT_NUM_SAMPLES}
}

FEATURE_SAMPLING_VALUES: Dict[str, Set[str]] = {
    SAMPLING_WITHOUT_REPLACEMENT: {ARGUMENT_SAMPLE_SIZE}
}

PARTITION_SAMPLING_VALUES: Dict[str, Set[str]] = {
    PARTITION_SAMPLING_RANDOM: {ARGUMENT_HOLDOUT_SET_SIZE},
    SAMPLING_STRATIFIED_LABEL_WISE: {ARGUMENT_HOLDOUT_SET_SIZE},
    SAMPLING_STRATIFIED_EXAMPLE_WISE: {ARGUMENT_HOLDOUT_SET_SIZE}
}

FEATURE_BINNING_VALUES: Dict[str, Set[str]] = {
    BINNING_EQUAL_FREQUENCY: {ARGUMENT_BIN_RATIO, ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS},
    BINNING_EQUAL_WIDTH: {ARGUMENT_BIN_RATIO, ARGUMENT_MIN_BINS, ARGUMENT_MAX_BINS}
}

PRUNING_VALUES: Set[str] = {PRUNING_IREP}


class SparsePolicy(Enum):
    AUTO = AUTOMATIC
    FORCE_SPARSE = 'sparse'
    FORCE_DENSE = 'dense'


class SparseFormat(Enum):
    CSC = 'csc'
    CSR = 'csr'


def create_sparse_policy(parameter_name: str, policy: str) -> SparsePolicy:
    try:
        return SparsePolicy(policy)
    except ValueError:
        raise ValueError('Invalid value given for parameter "' + parameter_name + '": Must be one of '
                         + format_enum_values(SparsePolicy) + ', but is "' + str(policy) + '"')


def create_label_sampling_factory(label_sampling: str) -> LabelSamplingFactory:
    if label_sampling is None:
        return NoLabelSamplingFactory()
    else:
        value, options = parse_param_and_options('label_sampling', label_sampling, LABEL_SAMPLING_VALUES)

        if value == SAMPLING_WITHOUT_REPLACEMENT:
            num_samples = options.get_int(ARGUMENT_NUM_SAMPLES, 1)
            return LabelSamplingWithoutReplacementFactory(num_samples)


def create_feature_sampling_factory(feature_sampling: str) -> FeatureSamplingFactory:
    if feature_sampling is None:
        return NoFeatureSamplingFactory()
    else:
        value, options = parse_param_and_options('feature_sampling', feature_sampling, FEATURE_SAMPLING_VALUES)

        if value == SAMPLING_WITHOUT_REPLACEMENT:
            sample_size = options.get_float(ARGUMENT_SAMPLE_SIZE, 0)
            return FeatureSamplingWithoutReplacementFactory(sample_size)


def create_partition_sampling_factory(holdout: str) -> PartitionSamplingFactory:
    if holdout is None:
        return NoPartitionSamplingFactory()
    else:
        value, options = parse_param_and_options('holdout', holdout, PARTITION_SAMPLING_VALUES)

        if value == PARTITION_SAMPLING_RANDOM:
            holdout_set_size = options.get_float(ARGUMENT_HOLDOUT_SET_SIZE, 0.33)
            return RandomBiPartitionSamplingFactory(holdout_set_size)
        if value == SAMPLING_STRATIFIED_LABEL_WISE:
            holdout_set_size = options.get_float(ARGUMENT_HOLDOUT_SET_SIZE, 0.33)
            return LabelWiseStratifiedBiPartitionSamplingFactory(holdout_set_size)
        if value == SAMPLING_STRATIFIED_EXAMPLE_WISE:
            holdout_set_size = options.get_float(ARGUMENT_HOLDOUT_SET_SIZE, 0.33)
            return ExampleWiseStratifiedBiPartitionSamplingFactory(holdout_set_size)


def create_pruning(pruning: str, instance_sampling: str) -> Pruning:
    if pruning is None:
        return NoPruning()
    else:
        value = parse_param('pruning', pruning, PRUNING_VALUES)

        if value == PRUNING_IREP:
            if instance_sampling is None:
                log.warning('Parameter "pruning" does not have any effect, because parameter "instance_sampling" is '
                            + 'set to "None"!')
                return NoPruning()
            return IREP()


def create_stopping_criteria(max_rules: int, time_limit: int) -> List[StoppingCriterion]:
    stopping_criteria: List[StoppingCriterion] = []

    if max_rules != 0:
        stopping_criteria.append(SizeStoppingCriterion(max_rules))

    if time_limit != 0:
        stopping_criteria.append(TimeStoppingCriterion(time_limit))

    return stopping_criteria


def get_preferred_num_threads(num_threads: int) -> int:
    if num_threads == 0:
        return os.cpu_count()
    return num_threads


def create_thresholds_factory(feature_binning: str, num_threads: int) -> ThresholdsFactory:
    if feature_binning is None:
        return ExactThresholdsFactory(num_threads)
    else:
        value, options = parse_param_and_options('feature_binning', feature_binning, FEATURE_BINNING_VALUES)

        if value == BINNING_EQUAL_FREQUENCY:
            bin_ratio = options.get_float(ARGUMENT_BIN_RATIO, 0.33)
            min_bins = options.get_int(ARGUMENT_MIN_BINS, 2)
            max_bins = options.get_int(ARGUMENT_MAX_BINS, 0)
            return ApproximateThresholdsFactory(EqualFrequencyFeatureBinning(bin_ratio, min_bins, max_bins),
                                                num_threads)
        elif value == BINNING_EQUAL_WIDTH:
            bin_ratio = options.get_float(ARGUMENT_BIN_RATIO, 0.33)
            min_bins = options.get_int(ARGUMENT_MIN_BINS, 2)
            max_bins = options.get_int(ARGUMENT_MAX_BINS, 0)
            return ApproximateThresholdsFactory(EqualWidthFeatureBinning(bin_ratio, min_bins, max_bins), num_threads)


def parse_param(parameter_name: str, value: str, allowed_values: Set[str]) -> str:
    if value in allowed_values:
        return value

    raise ValueError('Invalid value given for parameter "' + parameter_name + '": Must be one of '
                     + format_string_set(allowed_values) + ', but is "' + value + '"')


def parse_param_and_options(parameter_name: str, value: str,
                            allowed_values_and_options: Dict[str, Set[str]]) -> (str, Options):
    for allowed_value, allowed_options in allowed_values_and_options.items():
        if value.startswith(allowed_value):
            suffix = value[len(allowed_value):].strip()

            if len(suffix) > 0:
                try:
                    return allowed_value, Options.create(suffix, allowed_options)
                except ValueError as e:
                    raise ValueError('Invalid options specified for parameter "' + parameter_name + '" with value "'
                                     + allowed_value + '": ' + str(e))

            return allowed_value, Options()

    raise ValueError('Invalid value given for parameter "' + parameter_name + '": Must be one of '
                     + format_dict_keys(allowed_values_and_options) + ', but is "' + value + '"')


def should_enforce_sparse(m, sparse_format: SparseFormat, policy: SparsePolicy, dtype,
                          sparse_values: bool = True) -> bool:
    """
    Returns whether it is preferable to convert a given matrix into a `scipy.sparse.csr_matrix`,
    `scipy.sparse.csc_matrix` or `scipy.sparse.dok_matrix`, depending on the format of the given matrix and a given
    `SparsePolicy`:

    If the given policy is `SparsePolicy.AUTO`, the matrix will be converted into the given sparse format, if possible,
    if the sparse matrix is expected to occupy less memory than a dense matrix. To be able to convert the matrix into a
    sparse format, it must be a `scipy.sparse.lil_matrix`, `scipy.sparse.dok_matrix` or `scipy.sparse.coo_matrix`. If
    the given sparse format is `csr` or `csc` and the matrix is a already in that format, it will not be converted.

    If the given policy is `SparsePolicy.FORCE_DENSE`, the matrix will always be converted into the specified sparse
    format, if possible.

    If the given policy is `SparsePolicy.FORCE_SPARSE`, the matrix will always be converted into a dense matrix.

    :param m:               A `np.ndarray` or `scipy.sparse.matrix` to be checked
    :param sparse_format:   The `SparseFormat` to be used
    :param policy:          The `SparsePolicy` to be used
    :param dtype:           The type of the values that should be stored in the matrix
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
        x_sparse_policy = create_sparse_policy('feature_format', self.feature_format)
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
        y_sparse_policy = create_sparse_policy('label_format', self.label_format)
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
        rule_model_assemblage = self._create_rule_model_assemblage(num_labels)
        model_builder = self._create_model_builder()
        return rule_model_assemblage.induce_rules(nominal_feature_mask, feature_matrix, label_matrix, self.random_state,
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
        sparse_policy = create_sparse_policy('feature_format', self.feature_format)
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
    def _create_rule_model_assemblage(self, num_labels: int) -> RuleModelAssemblage:
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
