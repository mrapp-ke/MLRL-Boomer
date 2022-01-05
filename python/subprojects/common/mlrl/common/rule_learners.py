#!/usr/bin/python

"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for implementing single- or multi-label rule learning algorithms.
"""
from enum import Enum
from typing import Dict, Set

import numpy as np
from mlrl.common.data_types import DTYPE_UINT32
from mlrl.common.learners import Learner, NominalAttributeLearner
from mlrl.common.options import BooleanOption
from mlrl.common.options import Options
from mlrl.common.strings import format_enum_values, format_string_set, format_dict_keys
from scipy.sparse import issparse, isspmatrix_lil, isspmatrix_coo, isspmatrix_dok, isspmatrix_csc, isspmatrix_csr

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

ARGUMENT_NUM_THREADS = 'num_threads'

LABEL_SAMPLING_VALUES: Dict[str, Set[str]] = {
    SAMPLING_WITHOUT_REPLACEMENT: {ARGUMENT_NUM_SAMPLES}
}

FEATURE_SAMPLING_VALUES: Dict[str, Set[str]] = {
    SAMPLING_WITHOUT_REPLACEMENT: {ARGUMENT_SAMPLE_SIZE}
}

INSTANCE_SAMPLING_VALUES: Dict[str, Set[str]] = {
    SAMPLING_WITH_REPLACEMENT: {ARGUMENT_SAMPLE_SIZE},
    SAMPLING_WITHOUT_REPLACEMENT: {ARGUMENT_SAMPLE_SIZE},
    SAMPLING_STRATIFIED_LABEL_WISE: {ARGUMENT_SAMPLE_SIZE},
    SAMPLING_STRATIFIED_EXAMPLE_WISE: {ARGUMENT_SAMPLE_SIZE}
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

PARALLEL_VALUES: Dict[str, Set[str]] = {
    str(BooleanOption.TRUE.value): {ARGUMENT_NUM_THREADS},
    str(BooleanOption.FALSE.value): {}
}


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


def is_sparse(m, sparse_format: SparseFormat, dtype, sparse_values: bool = True) -> bool:
    """
    Returns whether a given matrix is considered sparse or not. A matrix is considered sparse if it is given in a sparse
    format and is expected to occupy less memory than a dense matrix.

    :param m:               A `np.ndarray` or `scipy.sparse.matrix` to be checked
    :param sparse_format:   The `SparseFormat` to be used
    :param dtype:           The type of the values that should be stored in the matrix
    :param sparse_values:   True, if the values must explicitly be stored when using a sparse format, False otherwise
    :return:                True, if the given matrix is considered sparse, False otherwise
    """
    if issparse(m):
        num_pointers = m.shape[1 if sparse_format == SparseFormat.CSC else 0]
        size_int = np.dtype(DTYPE_UINT32).itemsize
        size_data = np.dtype(dtype).itemsize if sparse_values else 0
        num_non_zero = m.nnz
        size_sparse = (num_non_zero * size_data) + (num_non_zero * size_int) + (num_pointers * size_int)
        size_dense = np.prod(m.shape) * size_data
        return size_sparse < size_dense
    return False


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
            return is_sparse(m, sparse_format=sparse_format, dtype=dtype, sparse_values=sparse_values)
        else:
            return policy == SparsePolicy.FORCE_SPARSE

    raise ValueError(
        'Matrix of type ' + type(m).__name__ + ' cannot be converted to format \'' + str(sparse_format) + '\'')


class MLRuleLearner(Learner, NominalAttributeLearner):
    """
    A scikit-multilearn implementation of a rule learning algorithm for multi-label classification or ranking.
    """

    def __init__(self, random_state: int, feature_format: str, label_format: str, prediction_format: str):
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

    def _fit(self, x, y):
        # TODO
        pass

    def _predict(self, x):
        # TODO
        pass
