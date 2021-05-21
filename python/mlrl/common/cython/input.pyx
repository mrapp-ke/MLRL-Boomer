"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_unique, make_shared
from libcpp.utility cimport move


cdef class LabelMatrix:
    """
    A wrapper for the pure virtual C++ class `ILabelMatrix`.
    """
    pass


cdef class CContiguousLabelMatrix(LabelMatrix):
    """
    A wrapper for the C++ class `CContiguousLabelMatrix`.
    """

    def __cinit__(self, const uint8[:, ::1] array):
        """
        :param array: A C-contiguous array of type `uint8`, shape `(num_examples, num_labels)`, that stores the labels
                      of the training examples
        """
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_labels = array.shape[1]
        self.label_matrix_ptr = <shared_ptr[ILabelMatrix]>make_shared[CContiguousLabelMatrixImpl](num_examples,
                                                                                                  num_labels,
                                                                                                  &array[0, 0])


cdef class CsrLabelMatrix(LabelMatrix):
    """
    A wrapper for the C++ class `CsrLabelMatrix`.
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
        self.label_matrix_ptr = <shared_ptr[ILabelMatrix]>make_shared[CsrLabelMatrixImpl](num_examples, num_labels,
                                                                                          &row_indices[0],
                                                                                          &col_indices[0])


cdef class FeatureMatrix:
    """
    A wrapper for the pure virtual C++ class `IFeatureMatrix`.
    """
    pass


cdef class FortranContiguousFeatureMatrix(FeatureMatrix):
    """
    A wrapper for the C++ class `FortranContiguousFeatureMatrix`.
    """

    def __cinit__(self, const float32[::1, :] array):
        """
        :param array: A Fortran-contiguous array of type `float32`, shape `(num_examples, num_features)`, that stores
                      the feature values of the training examples
        """
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_features = array.shape[1]
        self.feature_matrix_ptr = <shared_ptr[IFeatureMatrix]>make_shared[FortranContiguousFeatureMatrixImpl](
            num_examples, num_features, &array[0, 0])


cdef class CscFeatureMatrix(FeatureMatrix):
    """
    A wrapper for the C++ class `CscFeatureMatrix`.
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
        self.feature_matrix_ptr = <shared_ptr[IFeatureMatrix]>make_shared[CscFeatureMatrixImpl](num_examples,
                                                                                                num_features, &data[0],
                                                                                                &row_indices[0],
                                                                                                &col_indices[0])


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
        self.feature_matrix_ptr = make_shared[CContiguousFeatureMatrixImpl](num_examples, num_features, &array[0, 0])


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
        self.feature_matrix_ptr = make_shared[CsrFeatureMatrixImpl](num_examples, num_features, &data[0],
                                                                    &row_indices[0], &col_indices[0])


cdef class NominalFeatureMask:
    """
    A wrapper for the pure virtual C++ class `INominalFeatureMask`.
    """
    pass


cdef class BitNominalFeatureMask(NominalFeatureMask):
    """
    A wrapper for the C++ class `BitNominalFeatureMask`.
    """

    def __cinit__(self, uint32 num_features, list nominal_feature_indices):
        """
        :param num_features:            The total number of available features
        :param nominal_feature_indices: A list which contains the indices of all nominal features
        """
        cdef unique_ptr[BitNominalFeatureMaskImpl] ptr = make_unique[BitNominalFeatureMaskImpl](num_features)
        cdef uint32 i

        for i in nominal_feature_indices:
            ptr.get().setNominal(i)

        self.nominal_feature_mask_ptr = <shared_ptr[INominalFeatureMask]>move(ptr)


cdef class EqualNominalFeatureMask(NominalFeatureMask):
    """
    A wrapper for the C++ class `EqualNominalFeatureMask`.
    """

    def __cinit__(self, bint nominal):
        """
        :param nominal: True, if all features are nominal, false, if all features are not nominal
        """
        self.nominal_feature_mask_ptr = <shared_ptr[INominalFeatureMask]>make_shared[EqualNominalFeatureMaskImpl](
            nominal)
