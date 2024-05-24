"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for handling arrays.
"""
from enum import Enum
from typing import Optional, Set

import numpy as np

from scipy.sparse import issparse, isspmatrix_coo, isspmatrix_csc, isspmatrix_csr, isspmatrix_dok, isspmatrix_lil, \
    sparray

from mlrl.common.data_types import Uint32
from mlrl.common.format import format_iterable


class SparseFormat(Enum):
    """
    Specifies all valid textual representations of sparse matrix formats.
    """
    LIL = 'lil'
    COO = 'coo'
    DOK = 'dok'
    CSC = 'csc'
    CSR = 'csr'


def is_lil(array) -> bool:
    """
    Returns whether a given `scipy.sparse.spmatrix` or `scipy.sparse.sparray` uses the LIL format or not.

    :param array:   A `scipy.sparse.spmatrix` or `scipy.sparse.sparray` to be checked
    :return:        True, if the given array uses the LIL format, False otherwise
    """
    return isspmatrix_lil(array) or (isinstance(array, sparray) and array.format == 'lil')


def is_coo(array) -> bool:
    """
    Returns whether a given `scipy.sparse.spmatrix` or `scipy.sparse.sparray` uses the COO format or not.

    :param array:   A `scipy.sparse.spmatrix` or `scipy.sparse.sparray` to be checked
    :return:        True, if the given array uses the COO format, False otherwise
    """
    return isspmatrix_coo(array) or (isinstance(array, sparray) and array.format == 'coo')


def is_dok(array) -> bool:
    """
    Returns whether a given `scipy.sparse.spmatrix` or `scipy.sparse.sparray` uses the DOK format or not.

    :param array:   A `scipy.sparse.spmatrix` or `scipy.sparse.sparray` to be checked
    :return:        True, if the given array uses the DOK format, False otherwise
    """
    return isspmatrix_dok(array) or (isinstance(array, sparray) and array.format == 'dok')


def is_csc(array) -> bool:
    """
    Returns whether a given `scipy.sparse.spmatrix` or `scipy.sparse.sparray` uses the CSC format or not.

    :param array:   A `scipy.sparse.spmatrix` or `scipy.sparse.sparray` to be checked
    :return:        True, if the given array uses the CSC format, False otherwise
    """
    return isspmatrix_csc(array) or (isinstance(array, sparray) and array.format == 'csc')


def is_csr(array) -> bool:
    """
    Returns whether a given `scipy.sparse.spmatrix` or `scipy.sparse.sparray` uses the CSR format or not.

    :param array:   A `scipy.sparse.spmatrix` or `scipy.sparse.sparray` to be checked
    :return:        True, if the given array uses the CSR format, False otherwise
    """
    return isspmatrix_csr(array) or (isinstance(array, sparray) and array.format == 'csr')


def is_sparse(array, supported_formats: Optional[Set[SparseFormat]] = None) -> bool:
    """
    Returns whether a given array is a `scipy.sparse.spmatrix` or `scipy.sparse.sparray` or not.

    :param array:               A `np.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray` to be checked
    :param supported_formats:   A set of supported `SparseFormat`s, the `scipy.sparse.spmatrix` or
                                `scipy.sparse.sparray` may use or None, if the format should not be checked
    :return:                    True, if the given array is a `scipy.sparse.spmatrix` or `scipy.sparse.sparray` using
                                one of the supported formats, False otherwise
    """
    if supported_formats and len(supported_formats) > 0:
        lil = SparseFormat.LIL in supported_formats and is_lil(array)
        coo = SparseFormat.COO in supported_formats and is_coo(array)
        dok = SparseFormat.DOK in supported_formats and is_dok(array)
        csc = SparseFormat.CSC in supported_formats and is_csc(array)
        csr = SparseFormat.CSR in supported_formats and is_csr(array)

        if lil or coo or dok or csc or csr:
            return True
        return False

    return issparse(array)


def is_sparse_and_memory_efficient(array, sparse_format: SparseFormat, dtype, sparse_values: bool = True) -> bool:
    """
    Returns whether a given matrix uses sparse format and is expected to occupy less memory than a dense matrix.

    :param array:           A `np.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray` to be checked
    :param sparse_format:   The `SparseFormat` to be used. Must be `SparseFormat.CSC` or `SparseFormat.CSR`
    :param dtype:           The type of the values that should be stored in the matrix
    :param sparse_values:   True, if the values must explicitly be stored when using a sparse format, False otherwise
    :return:                True, if the given matrix uses a sparse format an is expected to occupy less memory than a
                            dense matrix, False otherwise
    """
    supported_formats = {SparseFormat.CSC, SparseFormat.CSR}

    if sparse_format not in supported_formats:
        raise ValueError('Unable to estimate memory requirements of given sparse format: Must be one of '
                         + format_iterable(supported_formats) + ', but is "' + str(sparse_format) + '"')

    if is_sparse(array):
        num_pointers = array.shape[1 if sparse_format == SparseFormat.CSC else 0]
        size_int = np.dtype(Uint32).itemsize
        size_data = np.dtype(dtype).itemsize
        size_sparse_data = size_data if sparse_values else 0
        num_dense_elements = array.nnz
        size_sparse = (num_dense_elements * size_sparse_data) + (num_dense_elements * size_int) + (num_pointers
                                                                                                   * size_int)
        size_dense = np.prod(array.shape) * size_data
        return size_sparse < size_dense
    return False


def enforce_dense(array, order: str, dtype, sparse_value=0) -> np.ndarray:
    """
    Converts a given array into a `np.ndarray`, if necessary, and enforces a specific memory layout and data type to be
    used.

    :param array:           A `np.ndarray`, `scipy.sparse.spmatrix` or `scipy.sparse.sparray` to be converted
    :param order:           The memory layout to be used. Must be `C` or `F`
    :param dtype:           The data type to be used
    :param sparse_value:    The value that should be used for sparse elements in the given array
    :return:                A `np.ndarray` that uses the given memory layout and data type
    """
    if is_sparse(array):
        if sparse_value != 0:
            dense_array = np.full(shape=array.shape, fill_value=sparse_value, dtype=dtype, order=order)
            dense_array[array.nonzero()] = 0
            dense_array += array
            return np.asarray(dense_array, dtype=dtype, order=order)
        return np.require(array.toarray(order=order), dtype=dtype)
    return np.require(array, dtype=dtype, requirements=[order])


def enforce_2d(array: np.ndarray) -> np.ndarray:
    """
    Converts a given `np.ndarray` into a two-dimensional array if it is one-dimensional.

    :param array:   A `np.ndarray` to be converted
    :return:        A `np.ndarray` with at least two dimensions
    """
    if array.ndim == 1:
        return np.expand_dims(array, axis=1)
    return array
