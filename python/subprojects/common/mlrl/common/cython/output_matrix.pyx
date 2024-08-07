"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class OutputMatrix:
    """
    An output matrix that stores the ground truth to be used for training a model.
    """

    cdef IOutputMatrix* get_output_matrix_ptr(self):
        pass

    def get_num_rows(self) -> int:
        """
        Returns the number of examples in the output matrix.

        :return The number of examples
        """
        return self.get_output_matrix_ptr().getNumExamples()

    def get_num_cols(self) -> int:
        """
        Returns the number of outputs in the output matrix.

        :return The number of outputs
        """
        return self.get_output_matrix_ptr().getNumOutputs()

    def is_sparse(self) -> bool:
        """
        Returns whether the output matrix is sparse or not.

        :return: True, if the output matrix is sparse, False otherwise
        """
        return self.get_output_matrix_ptr().isSparse()
