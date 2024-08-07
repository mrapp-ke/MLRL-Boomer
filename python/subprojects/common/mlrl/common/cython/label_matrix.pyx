"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class RowWiseLabelMatrix(OutputMatrix):
    """
    A label matrix that provides row-wise access to the labels of examples.
    """

    cdef IRowWiseLabelMatrix* get_row_wise_label_matrix_ptr(self):
        pass

    def calculate_label_cardinality(self) -> float:
        return self.get_row_wise_label_matrix_ptr().calculateLabelCardinality()


cdef class CContiguousLabelMatrix(RowWiseLabelMatrix):
    """
    A label matrix that provides row-wise access to the labels of examples that are stored in a C-contiguous array.
    """

    def __cinit__(self, const uint8[:, ::1] array not None):
        """
        :param array: A C-contiguous array of type `uint8`, shape `(num_examples, num_labels)`, that stores the labels
                      of the training examples
        """
        self.array = array
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_labels = array.shape[1]
        self.label_matrix_ptr = createCContiguousLabelMatrix(&array[0, 0], num_examples, num_labels)

    cdef IOutputMatrix* get_output_matrix_ptr(self):
        return self.label_matrix_ptr.get()

    cdef IRowWiseLabelMatrix* get_row_wise_label_matrix_ptr(self):
        return self.label_matrix_ptr.get()


cdef class CsrLabelMatrix(RowWiseLabelMatrix):
    """
    A label matrix that provides row-wise access to the labels of examples that are stored in a sparse matrix in the
    compressed sparse row (CSR) format.
    """

    def __cinit__(self, uint32[::1] indices not None, uint32[::1] indptr not None, uint32 num_examples,
                  uint32 num_labels):
        """
        :param indices:         An array of type `uint32`, shape `(num_dense_elements)`, that stores the column-indices
                                of all dense elements explicitly stored in the matrix
        :param indptr:          An array of type `uint32`, shape `(num_examples + 1)`, that stores the indices of the
                                first element in `indices` that corresponds to a certain example. The index at the last
                                position is equal to `num_dense_elements`
        :param num_examples:    The total number of examples
        :param num_labels:      The total number of labels
        """
        self.indices = indices
        self.indptr = indptr
        self.label_matrix_ptr = createCsrLabelMatrix(&indices[0], &indptr[0], num_examples, num_labels)

    cdef IOutputMatrix* get_output_matrix_ptr(self):
        return self.label_matrix_ptr.get()

    cdef IRowWiseLabelMatrix* get_row_wise_label_matrix_ptr(self):
        return self.label_matrix_ptr.get()
