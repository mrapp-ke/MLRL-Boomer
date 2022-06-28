"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for implementing single- or multi-label rule learning algorithms.
"""
import logging as log
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np
from mlrl.common.arrays import enforce_dense
from mlrl.common.cython.feature_matrix import FortranContiguousFeatureMatrix, CscFeatureMatrix, CsrFeatureMatrix, \
    CContiguousFeatureMatrix
from mlrl.common.cython.label_matrix import CContiguousLabelMatrix, CsrLabelMatrix
from mlrl.common.cython.learner import RuleLearner as RuleLearnerWrapper
from mlrl.common.cython.nominal_feature_mask import EqualNominalFeatureMask, MixedNominalFeatureMask
from mlrl.common.data_types import DTYPE_UINT8, DTYPE_UINT32, DTYPE_FLOAT32
from mlrl.common.learners import Learner, NominalAttributeLearner
from mlrl.common.strings import format_enum_values
from scipy.sparse import issparse, isspmatrix_lil, isspmatrix_coo, isspmatrix_dok, isspmatrix_csc, isspmatrix_csr
from sklearn.utils import check_array


class SparsePolicy(Enum):
    AUTO = 'auto'
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


def get_int(value) -> Optional[int]:
    return int(value) if value is not None else None


def get_float(value) -> Optional[float]:
    return float(value) if value is not None else None


def get_string(value) -> Optional[str]:
    return str(value) if value is not None else None


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
        size_data = np.dtype(dtype).itemsize
        size_sparse_data = size_data if sparse_values else 0
        num_non_zero = m.nnz
        size_sparse = (num_non_zero * size_sparse_data) + (num_non_zero * size_int) + (num_pointers * size_int)
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
    the given sparse format is `csr` or `csc` and the matrix is already in that format, it will not be converted.

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
        'Matrix of type ' + type(m).__name__ + ' cannot be converted to format "' + str(sparse_format) + '""')


class RuleLearner(Learner, NominalAttributeLearner, ABC):
    """
    A scikit-learn implementation of a rule learning algorithm for multi-label classification or ranking.
    """

    def __init__(self, random_state: int, feature_format: str, label_format: str, predicted_label_format: str):
        """
        :param random_state:            The seed to be used by RNGs. Must be at least 1
        :param feature_format:          The format to be used for the representation of the feature matrix. Must be
                                        `sparse`, `dense` or `auto`
        :param label_format:            The format to be used for the representation of the label matrix. Must be
                                        `sparse`, `dense` or 'auto'
        :param predicted_label_format:  The format to be used for representation of predicted labels. Must be `sparse`,
                                        `dense` or `auto`
        """
        super().__init__()
        self.random_state = random_state
        self.feature_format = feature_format
        self.label_format = label_format
        self.predicted_label_format = predicted_label_format

    def _fit(self, x, y):
        # Validate feature matrix and convert it to the preferred format...
        x_sparse_format = SparseFormat.CSC
        x_sparse_policy = create_sparse_policy('feature_format', self.feature_format)
        x_enforce_sparse = should_enforce_sparse(x, sparse_format=x_sparse_format, policy=x_sparse_policy,
                                                 dtype=DTYPE_FLOAT32)
        x = self._validate_data((x if x_enforce_sparse else enforce_dense(x, order='F', dtype=DTYPE_FLOAT32)),
                                accept_sparse=(x_sparse_format.value if x_enforce_sparse else False),
                                dtype=DTYPE_FLOAT32, force_all_finite='allow-nan')

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

        # Check if predictions should be sparse...
        prediction_sparse_policy = create_sparse_policy('predicted_label_format', self.predicted_label_format)
        self.sparse_predictions_ = prediction_sparse_policy != SparsePolicy.FORCE_DENSE and (
                prediction_sparse_policy == SparsePolicy.FORCE_SPARSE or
                is_sparse(y, sparse_format=y_sparse_format, dtype=DTYPE_UINT8, sparse_values=False))

        y_sparse_policy = create_sparse_policy('label_format', self.label_format)
        y_enforce_sparse = should_enforce_sparse(y, sparse_format=y_sparse_format, policy=y_sparse_policy,
                                                 dtype=DTYPE_UINT8, sparse_values=False)
        y = check_array((y if y_enforce_sparse else enforce_dense(y, order='C', dtype=DTYPE_UINT8)),
                        accept_sparse=(y_sparse_format.value if y_enforce_sparse else False), ensure_2d=False,
                        dtype=DTYPE_UINT8)

        if issparse(y):
            log.debug('A sparse matrix is used to store the labels of the training examples')
            y_row_indices = np.ascontiguousarray(y.indptr, dtype=DTYPE_UINT32)
            y_col_indices = np.ascontiguousarray(y.indices, dtype=DTYPE_UINT32)
            label_matrix = CsrLabelMatrix(y.shape[0], y.shape[1], y_row_indices, y_col_indices)
        else:
            log.debug('A dense matrix is used to store the labels of the training examples')
            label_matrix = CContiguousLabelMatrix(y)

        # Create a mask that provides access to the information whether individual features are nominal or not...
        num_features = feature_matrix.get_num_cols()

        if self.nominal_attribute_indices is None or len(self.nominal_attribute_indices) == 0:
            nominal_feature_mask = EqualNominalFeatureMask(False)
        elif len(self.nominal_attribute_indices) == num_features:
            nominal_feature_mask = EqualNominalFeatureMask(True)
        else:
            nominal_feature_mask = MixedNominalFeatureMask(num_features, self.nominal_attribute_indices)

        # Induce rules...
        learner = self._create_learner()
        training_result = learner.fit(nominal_feature_mask, feature_matrix, label_matrix, self.random_state)
        self.num_labels_ = training_result.num_labels
        self.label_space_info_ = training_result.label_space_info
        return training_result.rule_model

    def _predict(self, x):
        feature_matrix = self.__create_row_wise_feature_matrix(x)
        learner = self._create_learner()

        if self.sparse_predictions_:
            log.debug('A sparse matrix is used to store the predicted labels')
            return learner.predict_sparse_labels(feature_matrix, self.model_, self.label_space_info_, self.num_labels_)
        else:
            log.debug('A dense matrix is used to store the predicted labels')
            return learner.predict_labels(feature_matrix, self.model_, self.label_space_info_, self.num_labels_)

    def _predict_proba(self, x):
        learner = self._create_learner()
        feature_matrix = self.__create_row_wise_feature_matrix(x)
        num_labels = self.num_labels_

        if learner.can_predict_probabilities(feature_matrix, num_labels):
            log.debug('A dense matrix is used to store the predicted probability estimates')
            return learner.predict_probabilities(feature_matrix, self.model_, self.label_space_info_, num_labels)
        else:
            super()._predict_proba(x)

    def __create_row_wise_feature_matrix(self, x):
        sparse_format = SparseFormat.CSR
        sparse_policy = create_sparse_policy('feature_format', self.feature_format)
        enforce_sparse = should_enforce_sparse(x, sparse_format=sparse_format, policy=sparse_policy,
                                               dtype=DTYPE_FLOAT32)
        x = self._validate_data(x if enforce_sparse else enforce_dense(x, order='C', dtype=DTYPE_FLOAT32), reset=False,
                                accept_sparse=(sparse_format.value if enforce_sparse else False), dtype=DTYPE_FLOAT32,
                                force_all_finite='allow-nan')

        if issparse(x):
            log.debug('A sparse matrix is used to store the feature values of the query examples')
            x_data = np.ascontiguousarray(x.data, dtype=DTYPE_FLOAT32)
            x_row_indices = np.ascontiguousarray(x.indptr, dtype=DTYPE_UINT32)
            x_col_indices = np.ascontiguousarray(x.indices, dtype=DTYPE_UINT32)
            return CsrFeatureMatrix(x.shape[0], x.shape[1], x_data, x_row_indices, x_col_indices)
        else:
            log.debug('A dense matrix is used to store the feature values of the query examples')
            return CContiguousFeatureMatrix(x)

    @abstractmethod
    def _create_learner(self) -> RuleLearnerWrapper:
        """
        Must be implemented by subclasses in order to configure and create an implementation of the rule learner.

        :return: The implementation of the rule learner that has been created
        """
        pass
