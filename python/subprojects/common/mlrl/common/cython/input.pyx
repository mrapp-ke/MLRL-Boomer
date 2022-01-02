"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.utility cimport move
from libcpp.memory cimport make_unique


cdef class LabelMatrix:
    """
    A wrapper for the pure virtual C++ class `ILabelMatrix`.
    """

    cdef ILabelMatrix* get_label_matrix_ptr(self):
        pass

    def get_num_rows(self) -> int:
        """
        Returns the number of rows in the matrix.

        :return The number of rows
        """
        return self.get_label_matrix_ptr().getNumRows()

    def get_num_cols(self) -> int:
        """
        Returns the number of columns in the matrix.

        :return The number of columns
        """
        return self.get_label_matrix_ptr().getNumCols()

    def is_sparse(self) -> bool:
        """
        Returns whether the label matrix is sparse or not.

        :return: True, if the label matrix is sparse, False otherwise
        """
        return False


cdef class RowWiseLabelMatrix(LabelMatrix):
    """
    A wrapper for the pure virtual C++ class `IRowWiseLabelMatrix`.
    """

    cdef IRowWiseLabelMatrix* get_row_wise_label_matrix_ptr(self):
        pass

    def calculate_label_cardinality(self) -> int:
        return self.get_row_wise_label_matrix_ptr().calculateLabelCardinality()


cdef class CContiguousLabelMatrix(RowWiseLabelMatrix):
    """
    A wrapper for the pure virtual C++ class `ICContiguousLabelMatrix`.
    """

    def __cinit__(self, const uint8[:, ::1] array):
        """
        :param array: A C-contiguous array of type `uint8`, shape `(num_examples, num_labels)`, that stores the labels
                      of the training examples
        """
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_labels = array.shape[1]
        self.label_matrix_ptr = createCContiguousLabelMatrix(num_examples, num_labels, &array[0, 0])

    cdef ILabelMatrix* get_label_matrix_ptr(self):
        return self.label_matrix_ptr.get()

    cdef IRowWiseLabelMatrix* get_row_wise_label_matrix_ptr(self):
        return self.label_matrix_ptr.get()


cdef class CsrLabelMatrix(RowWiseLabelMatrix):
    """
    A wrapper for the pure virtual C++ class `ICsrLabelMatrix`.
    """

    def __cinit__(self, uint32 num_examples, uint32 num_labels, uint32[::1] row_indices, uint32[::1] col_indices):
        """
        :param num_examples:    The total number of examples
        :param num_labels:      The total number of labels
        :param row_indices:     An array of type `uint32`, shape `(num_examples + 1)`, that stores the indices of the
                                first element in `col_indices` that corresponds to a certain example. The index at the
                                last position is equal to `num_non_zero_values`
        :param col_indices:     An array of type `uint32`, shape `(num_non_zero_values)`, that stores the
                                column-indices, the relevant labels correspond to
        """
        self.label_matrix_ptr = createCsrLabelMatrix(num_examples, num_labels, &row_indices[0], &col_indices[0])

    cdef ILabelMatrix* get_label_matrix_ptr(self):
        return self.label_matrix_ptr.get()

    cdef IRowWiseLabelMatrix* get_row_wise_label_matrix_ptr(self):
        return self.label_matrix_ptr.get()

    def is_sparse(self) -> bool:
        return True


cdef class FeatureMatrix:
    """
    A wrapper for the pure virtual C++ class `IFeatureMatrix`.
    """

    cdef IFeatureMatrix* get_feature_matrix_ptr(self):
        pass

    def get_num_rows(self) -> int:
        """
        Returns the number of rows in the matrix.

        :return The number of rows
        """
        return self.get_feature_matrix_ptr().getNumRows()

    def get_num_cols(self) -> int:
        """
        Returns the number of columns in the matrix.

        :return The number of columns
        """
        return self.get_feature_matrix_ptr().getNumCols()

    def is_sparse(self) -> bool:
        """
        Returns whether the feature matrix is sparse or not.

        :return: True, if the feature matrix is sparse, False otherwise
        """
        return False


cdef class ColumnWiseFeatureMatrix(FeatureMatrix):
    """
    A wrapper for the pure virtual C++ class `IColumnWiseFeatureMatrix`.
    """

    cdef IColumnWiseFeatureMatrix* get_column_wise_feature_matrix_ptr(self):
        pass


cdef class FortranContiguousFeatureMatrix(ColumnWiseFeatureMatrix):
    """
    A wrapper for the pure virtual C++ class `IFortranContiguousFeatureMatrix`.
    """

    def __cinit__(self, const float32[::1, :] array):
        """
        :param array: A Fortran-contiguous array of type `float32`, shape `(num_examples, num_features)`, that stores
                      the feature values of the training examples
        """
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_features = array.shape[1]
        self.feature_matrix_ptr = createFortranContiguousFeatureMatrix(num_examples, num_features, &array[0, 0])

    cdef IFeatureMatrix* get_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()

    cdef IColumnWiseFeatureMatrix* get_column_wise_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()


cdef class CscFeatureMatrix(ColumnWiseFeatureMatrix):
    """
    A wrapper for the C++ class `ICscFeatureMatrix`.
    """

    def __cinit__(self, uint32 num_examples, uint32 num_features, const float32[::1] data, uint32[::1] row_indices,
                  uint32[::1] col_indices):
        """
        :param num_examples:    The total number of examples
        :param num_features:    The total number of features
        :param data:            An array of type `float32`, shape `(num_non_zero_values)`, that stores all non-zero
                                feature values
        :param row_indices:     An array of type `uint32`, shape `(num_non_zero_values)`, that stores the row-indices,
                                the values in `data` correspond to
        :param col_indices:     An array of type `uint32`, shape `(num_features + 1)`, that stores the indices of the
                                first element in `data` and `row_indices` that corresponds to a certain feature. The
                                index at the last position is equal to `num_non_zero_values`
        """
        self.feature_matrix_ptr = createCscFeatureMatrix(num_examples, num_features, &data[0], &row_indices[0],
                                                         &col_indices[0])

    cdef IFeatureMatrix* get_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()

    cdef IColumnWiseFeatureMatrix* get_column_wise_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()

    def is_sparse(self) -> bool:
        return True


cdef class CContiguousFeatureMatrix:
    """
    A wrapper for the C++ class `CContiguousFeatureMatrix`.
    """

    def __cinit__(self, const float32[:, ::1] array):
        """
        :param array: A C-contiguous array of type `float32`, shape `(num_examples, num_features)`, that stores the
                      feature values of the training examples
        """
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_features = array.shape[1]
        self.feature_matrix_ptr = make_unique[CContiguousFeatureMatrixImpl](num_examples, num_features, &array[0, 0])

    def get_num_rows(self) -> int:
        """
        Returns the number of rows in the matrix.

        :return The number of rows
        """
        return self.feature_matrix_ptr.get().getNumRows()

    def get_num_cols(self) -> int:
        """
        Returns the number of columns in the matrix.

        :return The number of columns
        """
        return self.feature_matrix_ptr.get().getNumCols()

    def is_sparse(self) -> bool:
        """
        Returns whether the feature matrix is sparse or not.

        :return: True, if the feature matrix is sparse, False otherwise
        """
        return False


cdef class CsrFeatureMatrix:
    """
    A wrapper for the C++ class `CsrFeatureMatrix`.
    """

    def __cinit__(self, uint32 num_examples, uint32 num_features, const float32[::1] data, uint32[::1] row_indices,
                  uint32[::1] col_indices):
        """
        :param num_examples:    The total number of examples
        :param num_features:    The total number of features
        :param data:            An array of type `float32`, shape `(num_non_zero_values)`, that stores all non-zero
                                feature values
        :param row_indices:     An array of type `uint32`, shape `(num_examples + 1)`, that stores the indices of the
                                first element in `data` and `col_indices` that corresponds to a certain example. The
                                index at the last position is equal to `num_non_zero_values`
        :param col_indices:     An array of type `uint32`, shape `(num_non_zero_values)`, that stores the
                                column-indices, the values in `data` correspond to
        """
        self.feature_matrix_ptr = make_unique[CsrFeatureMatrixImpl](num_examples, num_features, &data[0],
                                                                    &row_indices[0], &col_indices[0])

    def get_num_rows(self) -> int:
        """
        Returns the number of rows in the matrix.

        :return The number of rows
        """
        return self.feature_matrix_ptr.get().getNumRows()

    def get_num_cols(self) -> int:
        """
        Returns the number of columns in the matrix.

        :return The number of columns
        """
        return self.feature_matrix_ptr.get().getNumCols()

    def is_sparse(self) -> bool:
        """
        Returns whether the feature matrix is sparse or not.

        :return: True, if the feature matrix is sparse, False otherwise
        """
        return False

    def is_sparse(self) -> bool:
        return True


cdef class NominalFeatureMask:
    """
    A wrapper for the pure virtual C++ class `INominalFeatureMask`.
    """

    cdef INominalFeatureMask* get_nominal_feature_mask_ptr(self):
        pass


cdef class EqualNominalFeatureMask(NominalFeatureMask):
    """
    A wrapper for the pure virtual C++ class `IEqualNominalFeatureMask`.
    """

    def __cinit__(self, bint nominal):
        """
        :param nominal: True, if all features are nominal, False, if all features are numerical/ordinal
        """
        self.nominal_feature_mask_ptr = createEqualNominalFeatureMask(nominal)

    cdef INominalFeatureMask* get_nominal_feature_mask_ptr(self):
        return self.nominal_feature_mask_ptr.get()


cdef class MixedNominalFeatureMask(NominalFeatureMask):
    """
    A wrapper for the pure virtual C++ class `IMixedNominalFeatureMask`.
    """

    def __cinit__(self, uint32 num_features, list nominal_feature_indices):
        """
        :param num_features:            The total number of available features
        :param nominal_feature_indices: A list which contains the indices of all nominal features
        """
        cdef unique_ptr[IMixedNominalFeatureMask] nominal_feature_mask_ptr = createMixedNominalFeatureMask(num_features)
        cdef uint32 i

        for i in nominal_feature_indices:
            nominal_feature_mask_ptr.get().setNominal(i, True)

        self.nominal_feature_mask_ptr = move(nominal_feature_mask_ptr)

    cdef INominalFeatureMask* get_nominal_feature_mask_ptr(self):
        return self.nominal_feature_mask_ptr.get()

    def set_nominal(self, feature_index: int, bool nominal):
        self.nominal_feature_mask_ptr.get().setNominal(feature_index, nominal)
