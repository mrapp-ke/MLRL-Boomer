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
    A base class for all classes that store statistics about the labels of the training examples, which serve as the
    basis for learning a new rule or refining an existing one.
    """

    cdef void reset_statistics(self):
        """
        Resets the statistics which should be considered in the following for learning a new rule or refining an
        existing one. The indices of the respective statistics must be provided via subsequent calls to the function
        `add_sampled_statistic` or `update_covered_statistic`.

        This function must be invoked before a new rule is learned from scratch (as each rule may be learned on a
        different sub-sample of the statistics), as well as each time an existing rule has been refined, i.e., when a
        new condition has been added to its body (because this results in a subset of the statistics being covered by
        the refined rule).

        This function is supposed to reset any non-global internal state that only holds for a certain subset of the
        available statistics and therefore becomes invalid when a different subset of the statistics should be used.
        """
        pass

    cdef void add_sampled_statistic(self, intp statistic_index, uint32 weight):
        """
        Adds a specific statistic to the sub-sample that should be considered in the following for learning a new rule
        from scratch.

        This function must be called repeatedly for each statistic that should be considered, immediately after the
        invocation of the function `reset_statistics`.

        This function is supposed to update any internal state that relates to the considered statistics, i.e., to
        compute and store local information that is required by the other function that will be called later. Any
        information computed by this function is expected to be reset when invoking the function `reset_statistics` for
        the next time.

        :param statistic_index: The index of the statistic that should be considered
        :param weight:          The weight of the statistic that should be considered
        """
        pass

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove):
        """
        Adds a specific statistic to the subset that is covered by an existing rule and therefore should be considered
        in the following for refining an existing rule.

        This function must be called repeatedly for each statistic that is covered by the existing rule, immediately
        after the invocation of the function `reset_statistics`.

        Alternatively, this function may be used to indicate that a statistic, which has previously been passed to this
        function, should not be considered anymore by setting the argument `remove` accordingly.

        This function is supposed to update any internal state that relates to the considered statistics, i.e., to
        compute and store local information that is required by the other function that will be called later. Any
        information computed by this function is expected to be reset when invoking the function `reset_statistics` for
        the next time.

        :param statistic_index: The index of the statistic that should be updated
        :param weight:          The weight of the statistic that should be updated
        :param remove:          0, if the statistic should be considered, 1, if the statistic should not be considered
                                anymore
        """
        pass

    cdef RefinementSearch begin_search(self, intp[::1] label_indices):
        """
        Starts a new search for the best refinement of a rule. The statistics that are covered by such a refinement must
        be provided via subsequent calls to the function `RefinementSearch#update_search`.

        This function must be called each time a new refinement is considered, unless the refinement covers all
        statistics previously provided via calls to the function `RefinementSearch#update_search`.

        Optionally, a subset of the available labels may be specified via the argument `label_indices`. In such case,
        only the specified labels will be considered by the search. When calling this function again to start another
        search from scratch, a different set of labels may be specified.

        :param label_indices:   An array of dtype `intp`, shape `(num_considered_labels)`, representing the indices of
                                the labels that should be considered by the search or None, if all labels should be
                                considered
        :return:                A new object of type `RefinementSearch` to be used to conduct the search
        """
        pass

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head):
        """
        Updates a specific statistic based on the predictions of a newly induced rule.

        This function must be called for each statistic that is covered by the new rule before learning the next rule,
        i.e., prior to the next invocation of the function `reset_statistics`.

        :param statistic_index: The index of the statistic to be updated
        :param label_indices:   An array of dtype `intp`, shape `(head.numPredictions_)`, representing the indices of
                                the labels for which the newly induced rule predicts or None, if the rule predicts for
                                all labels
        :param head:            A pointer to an object of type `HeadCandidate`, representing the head of the newly
                                induced rule
        """
        pass
