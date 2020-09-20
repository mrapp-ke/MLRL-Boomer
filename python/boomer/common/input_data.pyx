"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that provide access to the data that is provided for training.
"""
from libcpp.memory cimport make_shared


cdef class LabelMatrix:
    """
    A wrapper for the abstract C++ class `AbstractLabelMatrix`.
    """
    pass


cdef class RandomAccessLabelMatrix(LabelMatrix):
    """
    A wrapper for the abstract C++ class `AbstractRandomAccessLabelMatrix`.
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
        self.label_matrix_ptr = <shared_ptr[AbstractLabelMatrix]>make_shared[DenseLabelMatrixImpl](num_examples,
                                                                                                   num_labels, &y[0, 0])


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
        cdef shared_ptr[BinaryDokMatrix] dok_matrix_ptr = make_shared[BinaryDokMatrix](num_examples, num_labels)
        cdef uint32 num_rows = rows.shape[0]
        cdef list col_indices
        cdef uint32 r, c

        for r in range(num_rows):
            col_indices = rows[r]

            for c in col_indices:
                dok_matrix_ptr.get().set(r, c)

        self.label_matrix_ptr = <shared_ptr[AbstractLabelMatrix]>make_shared[DokLabelMatrixImpl](dok_matrix_ptr)


cdef class FeatureMatrix:
    """
    A wrapper for the abstract C++ class `AbstractFeatureMatrix`.
    """
    pass


cdef class DenseFeatureMatrix(FeatureMatrix):
    """
    A wrapper for the C++ class `DenseFeatureMatrixImpl`.
    """

    def __cinit__(self, const float32[::1, :] x):
        """
        :param x: An array of type `float32`, shape `(num_examples, num_features)`, representing the feature values of
                  the training examples
        """
        cdef uint32 num_examples = x.shape[0]
        cdef uint32 num_features = x.shape[1]
        self.feature_matrix_ptr = <shared_ptr[AbstractFeatureMatrix]>make_shared[DenseFeatureMatrixImpl](num_examples,
                                                                                                         num_features,
                                                                                                         &x[0, 0])


cdef class CscFeatureMatrix(FeatureMatrix):
    """
    A wrapper for the C++ class `CscFeatureMatrixImpl`.
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
        self.feature_matrix_ptr = <shared_ptr[AbstractFeatureMatrix]>make_shared[CscFeatureMatrixImpl](
            num_examples, num_features, &x_data[0], &x_row_indices[0], &x_col_indices[0])


cdef class NominalFeatureSet:
    """
    A wrapper for the C++ class `AbstractNominalFeatureSet`.
    """
    pass


cdef class DokNominalFeatureSet(NominalFeatureSet):
    """
    A wrapper for the C++ class `DokNominalFeatureSetImpl`.
    """

    """
    :param nominal_feature_indices: A list which contains the indices of all nominal features
    """
    def __cinit__(self, list nominal_feature_indices):
        cdef uint32 num_nominal_features = len(nominal_feature_indices)
        cdef shared_ptr[BinaryDokVector] dok_vector_ptr = make_shared[BinaryDokVector](num_nominal_features)
        cdef uint32 i

        for i in nominal_feature_indices:
            dok_vector_ptr.get().set(i)

        self.nominal_feature_set_ptr = <shared_ptr[AbstractNominalFeatureSet]>make_shared[DokNominalFeatureSetImpl](
            dok_vector_ptr)
