"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class RowWiseRegressionMatrix(OutputMatrix):
    """
    A regression matrix that provides row-wise access to the regression scores of examples.
    """

    cdef IRowWiseRegressionMatrix* get_row_wise_regression_matrix_ptr(self):
        pass


cdef class CContiguousRegressionMatrix(RowWiseRegressionMatrix):
    """
    A regression matrix that provides row-wise access to the regression scores of examples that are stored in a
    C-contiguous array.
    """

    def __cinit__(self, const float32[:, ::1] array not None):
        """
        :param array: A C-contiguous array of type `float32`, shape `(num_examples, num_outputs)`, that stores the
                      regression scores of the training examples
        """
        self.array = array
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_outputs = array.shape[1]
        self.regression_matrix_ptr = createCContiguousRegressionMatrix(&array[0, 0], num_examples, num_outputs)

    cdef IOutputMatrix* get_output_matrix_ptr(self):
        return self.regression_matrix_ptr.get()
    
    cdef IRowWiseRegressionMatrix* get_row_wise_regression_matrix_ptr(self):
        return self.regression_matrix_ptr.get()


cdef class CsrRegressionMatrix(RowWiseRegressionMatrix):
    """
    A regression matrix that provides row-wise access to the regression scores of examples that are stored in a sparse
    matrix in the compressed sparse row (CSR) format.
    """

    def __cinit__(self, float32[::1] values, uint32[::1] indices not None, uint32[::1] indptr not None,
                  uint32 num_examples, uint32 num_outputs):
        """
        :param values:          An array of type `float32`, shape `(num_dense_elements)`, that stores the values of all
                                dense elements explicitly stored in the matrix
        :param indices:         An array of type `uint32`, shape `(num_dense_elements)`, that stores the column-indices
                                of all dense elements explicitly stored in the matrix
        :param indptr:          An array of type `uint32`, shape `(num_examples + 1)`, that stores the indices of the
                                first element in `indices` that corresponds to a certain example. The index at the last
                                position is equal to `num_dense_elements`
        :param num_examples:    The total number of examples
        :param num_outputs:     The total number of outputs
        """
        self.values = values
        self.indices = indices
        self.indptr = indptr
        self.regression_matrix_ptr = createCsrRegressionMatrix(&values[0], &indices[0], &indptr[0], num_examples,
                                                               num_outputs)

    cdef IOutputMatrix* get_output_matrix_ptr(self):
        return self.regression_matrix_ptr.get()

    cdef IRowWiseRegressionMatrix* get_row_wise_regression_matrix_ptr(self):
        return self.regression_matrix_ptr.get()
