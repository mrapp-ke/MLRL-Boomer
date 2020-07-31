"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that provide access to the data that is provided for training.
"""
from boomer.common._arrays cimport uint32


cdef class LabelMatrix:
    """
    A wrapper for the abstract C++ class `AbstractLabelMatrix`.
    """

    cdef uint8 get_label(self, intp example_index, intp label_index) nogil:
        """
        Returns whether a specific label of the example at a given index is relevant or irrelevant.

        :param example_index:   The index of the example
        :param label_index:     The index of the label
        :return:                1, if the label is relevant, 0 otherwise
        """
        cdef AbstractLabelMatrix* label_matrix = self.label_matrix
        return label_matrix.getLabel(example_index, label_index)


cdef class DenseLabelMatrix(LabelMatrix):
    """
    A wrapper for the C++ class `DenseLabelMatrix`.
    """

    def __cinit__(self, const uint8[:, ::1] y):
        """
        :param y: An array of dtype uint8, shape `(num_examples, num_labels)`, representing the labels of the training
                  examples
        """
        cdef intp num_examples = y.shape[0]
        cdef intp num_labels = y.shape[1]
        self.label_matrix = new DenseLabelMatrixImpl(num_examples, num_labels, &y[0, 0])
        self.num_examples = num_examples
        self.num_labels = num_labels

    def __dealloc__(self):
        del self.label_matrix


cdef class DokLabelMatrix(LabelMatrix):
    """
    A wrapper for the C++ class `DokLabelMatrix`.
    """

    def __cinit__(self, intp num_examples, intp num_labels, list[::1] rows):
        """
        :param num_examples:    The total number of examples
        :param num_labels:      The total number of labels
        :param rows:            An array of dtype `list`, shape `(num_rows)`, storing a list for each example containing
                                the column indices of all non-zero labels
        """
        cdef BinaryDokMatrix* dok_matrix = new BinaryDokMatrix()
        cdef intp num_rows = rows.shape[0]
        cdef list col_indices
        cdef uint32 r, c

        for r in range(num_rows):
            col_indices = rows[r]

            for c in col_indices:
                dok_matrix.addValue(r, c)

        self.label_matrix = new DokLabelMatrixImpl(num_examples, num_labels, dok_matrix)
        self.num_examples = num_examples
        self.num_labels = num_labels

    def __dealloc__(self):
        del self.label_matrix

