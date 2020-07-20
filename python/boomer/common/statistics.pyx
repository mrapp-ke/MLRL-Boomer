"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for all classes that allow to store statistics about the labels of training examples.
"""


cdef class LabelMatrix:
    """
    A base class for all classes that provide random access to the labels of the training examples.
    """

    cdef uint8 get_label(self, intp example_index, intp label_index):
        """
        Returns whether a specific label of the example at a given index is relevant or irrelevant.

        :param example_index:   The index of the example
        :param label_index:     The index of the label
        :return:                1, if the label is relevant, 0 otherwise
        """
        pass


cdef class DenseLabelMatrix(LabelMatrix):
    """
    Implements random access to the labels of the training examples based on a dense label matrix.

    The label matrix must be given as a dense C-contiguous array.
    """

    def __cinit__(self, const uint8[:, ::1] y):
        """
        :param y: An array of dtype uint8, shape `(num_examples, num_labels)`, representing the labels of the training
                  examples
        """
        self.y = y
        self.num_examples = y.shape[0]
        self.num_labels = y.shape[1]

    cdef uint8 get_label(self, intp example_index, intp label_index):
        cdef const uint8[:, ::1] y = self.y
        return y[example_index, label_index]


cdef class SparseLabelMatrix(LabelMatrix):
    """
    Implements random access to the labels of the training examples based on a sparse label matrix.

    The label matrix must be given as a `scipy.sparse.lil_matrix` and will internally be converted to the dictionary of
    keys (DOK) format.
    """

    def __cinit__(self, intp num_examples, intp num_labels, list[::1] rows):
        """
        :param num_examples:    The total number of examples
        :param num_labels:      The total number of labels
        :param rows:            An array of dtype `list`, shape `(num_rows)`, storing a list for each example containing
                                the column indices of all non-zero labels
        """
        self.num_examples = num_examples
        self.num_labels = num_labels
        cdef BinaryDokMatrix* dok_matrix = new BinaryDokMatrix()
        cdef intp num_rows = rows.shape[0]
        cdef list col_indices
        cdef uint32 r, c

        for r in range(num_rows):
            col_indices = rows[r]

            for c in col_indices:
                dok_matrix.addValue(r, c)

        self.dok_matrix = dok_matrix

    def __dealloc__(self):
        del self.dok_matrix

    cdef uint8 get_label(self, intp example_index, intp label_index):
        cdef BinaryDokMatrix* dok_matrix = self.dok_matrix
        return dok_matrix.getValue(<uint32>example_index, <uint32>label_index)


cdef class RefinementSearch:
    """
    A base class for all classes that allow to search for the best refinement of a rule based on previously stored
    statistics.
    """

    cdef void update_search(self, intp statistic_index, uint32 weight):
        """
        TODO

        :param statistic_index:
        :param weight:
        """
        pass

    cdef void reset_search(self):
        """
        TODO
        """
        pass

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated):
        """
        TODO

        :param uncovered:
        :param accumulated:
        """
        pass

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated):
        """
        TODO

        :param uncovered:
        :param accumulated:
        """
        pass


cdef class DecomposableRefinementSearch(RefinementSearch):
    """
    A base class for all classes that allow to search for the best refinement of a rule based on previously stored
    statistics in the decomposable case, i.e., when the label-wise predictions are the same as the example-wise
    predictions.
    """

    cdef void update_search(self, intp statistic_index, uint32 weight):
        pass

    cdef void reset_search(self):
        pass

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated):
        pass

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated):
        # In the decomposable case, the example-wise predictions are the same as the label-wise predictions...
        return <Prediction*>self.calculate_label_wise_prediction(uncovered, accumulated)


cdef class NonDecomposableRefinementSearch(RefinementSearch):
    """
    A base class for all classes that allow to search for the best refinement of a rule based on previously stored
    statistics in the non-decomposable case, i.e., when the label-wise predictions are not the same as the example-wise
    predictions.
    """

    cdef void update_search(self, intp statistic_index, uint32 weight):
        pass

    cdef void reset_search(self):
        pass

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated):
        pass

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated):
        pass


cdef class Statistics:
    """
    A base class for all classes that store statistics about the labels of the training examples.
    """

    cdef void reset_examples(self):
        """
        TODO
        """
        pass

    cdef void add_sampled_statistic(self, intp statistic_index, uint32 weight):
        """
        TODO

        :param statistic_index:
        :param weight:
        """
        pass

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove):
        """
        TODO

        :param statistic_index:
        :param weight:
        :param remove:
        """
        pass

    cdef RefinementSearch begin_search(self, intp[::1] label_indices):
        """
        TODO

        :param label_indices:
        :return:
        """
        pass

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head):
        """
        TODO

        :param statistic_index:
        :param label_indices:
        :param head:
        """
        pass
