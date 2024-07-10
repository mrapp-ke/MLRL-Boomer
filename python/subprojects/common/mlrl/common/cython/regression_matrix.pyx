"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class CContiguousRegressionMatrix(OutputMatrix):
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
