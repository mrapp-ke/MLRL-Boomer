"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that provide access to the data that is provided for training.
"""
from libcpp.memory cimport make_unique, make_shared
from libcpp.utility cimport move


cdef class LabelMatrix:
    """
    A wrapper for the pure virtual C++ class `ILabelMatrix`.
    """
    pass


cdef class RandomAccessLabelMatrix(LabelMatrix):
    """
    A wrapper for the pure virtual C++ class `IRandomAccessLabelMatrix`.
    """
    pass


cdef class DenseLabelMatrix(RandomAccessLabelMatrix):
    """
    A wrapper for the C++ class `DenseLabelMatrix`.
    """

    def __cinit__(self, const uint8[:, ::1] y):
        """
        :param y: An array of type `uint8`, shape `(num_examples, num_labels)`, representing the labels of the training
                  examples
        """
        cdef uint32 num_examples = y.shape[0]
        cdef uint32 num_labels = y.shape[1]
        self.label_matrix_ptr = <shared_ptr[ILabelMatrix]>make_shared[DenseLabelMatrixImpl](num_examples, num_labels,
                                                                                            &y[0, 0])


cdef class DokLabelMatrix(RandomAccessLabelMatrix):
    """
    A wrapper for the C++ class `DokLabelMatrix`.
    """

    def __cinit__(self, uint32 num_examples, uint32 num_labels, list[::1] rows):
        """
        :param num_examples:    The total number of examples
        :param num_labels:      The total number of labels
        :param rows:            An array of type `list`, shape `(num_rows)`, storing a list for each example containing
                                the column indices of all non-zero labels
        """
        cdef unique_ptr[DokLabelMatrixImpl] ptr = make_unique[DokLabelMatrixImpl](num_examples, num_labels)
        cdef uint32 num_rows = rows.shape[0]
        cdef list col_indices
        cdef uint32 r, c

        for r in range(num_rows):
            col_indices = rows[r]

            for c in col_indices:
                ptr.get().setValue(r, c)

        self.label_matrix_ptr = <shared_ptr[ILabelMatrix]>move(ptr)


cdef class FeatureMatrix:
    """
    A wrapper for the pure virtual C++ class `IFeatureMatrix`.
    """
    pass


cdef class FortranContiguousFeatureMatrix(FeatureMatrix):
    """
    A wrapper for the C++ class `FortranContiguousFeatureMatrix`.
    """

    def __cinit__(self, const float32[::1, :] x):
        """
        :param x: A Fortran-contiguous array of type `float32`, shape `(num_examples, num_features)`, representing the
                  feature values of the training examples
        """
        cdef uint32 num_examples = x.shape[0]
        cdef uint32 num_features = x.shape[1]
        self.feature_matrix_ptr = <shared_ptr[IFeatureMatrix]>make_shared[FortranContiguousFeatureMatrixImpl](
            num_examples, num_features, &x[0, 0])


cdef class CscFeatureMatrix(FeatureMatrix):
    """
    A wrapper for the C++ class `CscFeatureMatrix`.
    """

    def __cinit__(self, uint32 num_examples, uint32 num_features, const float32[::1] x_data,
                  const uint32[::1] x_row_indices, const uint32[::1] x_col_indices):
        """
        :param num_examples:    The total number of examples
        :param num_features:    The total number of features
        :param x_data:          An array of type `float32`, shape `(num_non_zero_feature_values)`, representing the
                                non-zero feature values of the training examples
        :param x_row_indices:   An array of type `uint32`, shape `(num_non_zero_feature_values)`, representing the
                                row-indices of the examples, the values in `x_data` correspond to
        :param x_col_indices:   An array of type `uint32`, shape `(num_features + 1)`, representing the indices of the
                                first element in `x_data` and `x_row_indices` that corresponds to a certain feature. The
                                index at the last position is equal to `num_non_zero_feature_values`
        """
        self.feature_matrix_ptr = <shared_ptr[IFeatureMatrix]>make_shared[CscFeatureMatrixImpl](num_examples,
                                                                                                num_features,
                                                                                                &x_data[0],
                                                                                                &x_row_indices[0],
                                                                                                &x_col_indices[0])


cdef class NominalFeatureMask:
    """
    A wrapper for the pure virtual C++ class `INominalFeatureMask`.
    """
    pass


cdef class DokNominalFeatureMask(NominalFeatureMask):
    """
    A wrapper for the C++ class `DokNominalFeatureMask`.
    """

    """
    :param nominal_feature_indices: A list which contains the indices of all nominal features or None, if no nominal
                                    features are available
    """
    def __cinit__(self, list nominal_feature_indices):
        cdef uint32 num_nominal_features = 0 if nominal_feature_indices is None else len(nominal_feature_indices)
        cdef unique_ptr[DokNominalFeatureMaskImpl] ptr = make_unique[DokNominalFeatureMaskImpl]()
        cdef uint32 i

        if num_nominal_features > 0:
            for i in nominal_feature_indices:
                ptr.get().setNominal(i)

        self.nominal_feature_mask_ptr = <shared_ptr[INominalFeatureMask]>move(ptr)
