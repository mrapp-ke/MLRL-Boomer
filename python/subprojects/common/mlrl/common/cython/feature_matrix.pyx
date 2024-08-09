"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class FeatureMatrix:
    """
    A feature matrix.
    """

    cdef IFeatureMatrix* get_feature_matrix_ptr(self):
        pass

    def get_num_examples(self) -> int:
        """
        Returns the number of examples in the feature matrix.

        :return The number of examples
        """
        return self.get_feature_matrix_ptr().getNumExamples()

    def get_num_features(self) -> int:
        """
        Returns the number of features in the feature matrix.

        :return The number of features
        """
        return self.get_feature_matrix_ptr().getNumFeatures()

    def is_sparse(self) -> bool:
        """
        Returns whether the feature matrix is sparse or not.

        :return: True, if the feature matrix is sparse, False otherwise
        """
        return self.get_feature_matrix_ptr().isSparse()


cdef class ColumnWiseFeatureMatrix(FeatureMatrix):
    """
    A feature matrix that provides column-wise access to the feature values of examples.
    """

    cdef IColumnWiseFeatureMatrix* get_column_wise_feature_matrix_ptr(self):
        pass


cdef class FortranContiguousFeatureMatrix(ColumnWiseFeatureMatrix):
    """
    A feature matrix that provides column-wise access to the feature values of examples that are stored in a
    Fortran-contiguous array.
    """

    def __cinit__(self, const float32[::1, :] array not None):
        """
        :param array: A Fortran-contiguous array of type `float32`, shape `(num_examples, num_features)`, that stores
                      the feature values of the training examples
        """
        self.array = array
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_features = array.shape[1]
        self.feature_matrix_ptr = createFortranContiguousFeatureMatrix(&array[0, 0], num_examples, num_features)

    cdef IFeatureMatrix* get_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()

    cdef IColumnWiseFeatureMatrix* get_column_wise_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()


cdef class CscFeatureMatrix(ColumnWiseFeatureMatrix):
    """
    A feature matrix that provides column-wise access to the feature values of examples that are stored in a sparse
    matrix in the compressed sparse column (CSC) format.
    """

    def __cinit__(self, const float32[::1] values not None, uint32[::1] indices not None, uint32[::1] indptr not None,
                  uint32 num_examples, uint32 num_features, float32 sparse_value = 0.0):
        """
        :param values:          An array of type `float32`, shape `(num_dense_elements)`, that stores the values of all
                                dense elements explicitly stored in the matrix
        :param indices:         An array of type `uint32`, shape `(num_dense_elements)`, that stores the row-indices,
                                the values in `values` correspond to
        :param indptr:          An array of type `uint32`, shape `(num_features + 1)`, that stores the indices of the
                                first element in `values` and `indices` that corresponds to a certain feature. The index
                                at the last position is equal to `num_dense_elements`
        :param num_examples:    The total number of examples
        :param num_features:    The total number of features
        :param sparse_value:    The value that should be used for sparse elements in the feature matrix
        """
        self.values = values
        self.indices = indices
        self.indptr = indptr
        self.feature_matrix_ptr = createCscFeatureMatrix(&values[0], &indices[0], &indptr[0], num_examples,
                                                         num_features, sparse_value)

    cdef IFeatureMatrix* get_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()

    cdef IColumnWiseFeatureMatrix* get_column_wise_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()


cdef class RowWiseFeatureMatrix(FeatureMatrix):
    """
    A feature matrix that provides row-wise access to the feature values of examples.
    """

    cdef IRowWiseFeatureMatrix* get_row_wise_feature_matrix_ptr(self):
        pass


cdef class CContiguousFeatureMatrix(RowWiseFeatureMatrix):
    """
    A feature matrix that provides row-wise access to the feature values of examples that are stored in a C-contiguous
    array.
    """

    def __cinit__(self, const float32[:, ::1] array not None):
        """
        :param array: A C-contiguous array of type `float32`, shape `(num_examples, num_features)`, that stores the
                      feature values of the training examples
        """
        self.array = array
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_features = array.shape[1]
        self.feature_matrix_ptr = createCContiguousFeatureMatrix(&array[0, 0], num_examples, num_features)

    cdef IFeatureMatrix* get_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()

    cdef IRowWiseFeatureMatrix* get_row_wise_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()


cdef class CsrFeatureMatrix(RowWiseFeatureMatrix):
    """
    A feature matrix that provides row-wise access to the feature values of examples that are stored in a sparse matrix
    in the compressed sparse row (CSR) format.
    """

    def __cinit__(self, const float32[::1] values not None, uint32[::1] indices not None, uint32[::1] indptr not None,
                  uint32 num_examples, uint32 num_features, float32 sparse_value = 0.0):
        """
        :param values:          An array of type `float32`, shape `(num_dense_elements)`, that stores the values of all
                                dense elements explicitly stored in the matrix
        :param indices:         An array of type `uint32`, shape `(num_dense_elements)`, that stores the column-indices,
                                the values in `values` correspond to
        :param indptr:          An array of type `uint32`, shape `(num_examples + 1)`, that stores the indices of the
                                first element in `values` and `indices` that corresponds to a certain example. The index
                                at the last position is equal to `num_dense_elements`
        :param num_examples:    The total number of examples
        :param num_features:    The total number of features
        :param sparse_value:    The value that should be used for sparse elements in the feature matrix
        """
        self.values = values
        self.indices = indices
        self.indptr = indptr
        self.feature_matrix_ptr = createCsrFeatureMatrix(&values[0], &indices[0], &indptr[0], num_examples,
                                                         num_features, sparse_value)

    cdef IFeatureMatrix* get_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()

    cdef IRowWiseFeatureMatrix* get_row_wise_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()
