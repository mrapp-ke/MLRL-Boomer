"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for all (surrogate) loss functions to be minimized locally by the rules learned during training.
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
    A base class for all classes that allow to search for the best refinement of a rule according to a certain loss.
    """

    cdef void update_search(self, intp example_index, uint32 weight):
        """
        Notifies the search about an example that is covered by the condition that is currently considered for refining
        a rule.

        This function must be called repeatedly for each example that is covered by the current condition, immediately
        after the invocation of the function `Loss#begin_search`. Each of these examples must have been provided earlier
        via the function `Loss#update_sub_sample`.

        This function is supposed to update any internal state of the search that relates to the examples that are
        covered current condition, i.e., to compute and store local information that is required by the other functions
        that will be called later, e.g. statistics about the ground truth labels of the covered examples. Any
        information computed by this function is expected to be reset when invoking the function `reset_search` for the
        next time.

        :param example_index:   The index of the covered example
        :param weight:          The weight of the covered example
        """
        pass

    cdef void reset_search(self):
        """
        Resets the internal state of the search that has been updated by preceding calls to the `update_search` function
        to the state when the search was started via the function `Loss#begin_search`. When calling this function, the
        current state is not purged entirely, but it is cached and made available for use by the functions
        `calculate_example_wise_prediction` and `calculate_label_wise_prediction` (if the function argument
        `accumulated` is set accordingly).

        This function may be invoked multiple times (with one or several calls to `update_search` in between), which is
        supposed to update the previously cached state by accumulating the new one, i.e., the accumulated cached state
        should be the same as if `reset_search` would not have been called at all.
        """
        pass

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated):
        """
        Calculates and returns the loss-minimizing scores to be predicted by a rule that covers all examples that have
        been provided to the search so far via the function `update_search`.

        If the argument `uncovered` is 1, the rule is considered to cover all examples that belong to the difference
        between the examples that have been provided via the function `update_sub_sample` and the examples that have
        been provided via the function `update_search`.

        If the argument `accumulated` is 1, all examples that have been provided since the search has been started via
        the function `Loss#begin_search` are taken into account even if the function `reset_search` has been called
        before. If the latter has not been invoked, the argument does not have any effect.

        The calculated scores correspond to the subset of labels that has been provided when starting the search via the
        function `Loss#begin_search`. The score to be predicted for an individual label is calculated independently from
        the other labels, i.e., in case of a non-decomposable loss function, it is assumed that the rule will not
        predict for the other labels. In addition to each score, a quality score, which assesses the quality of the
        prediction for the respective label, is returned.

        :param uncovered:   0, if the rule covers all examples that have been provided via the function `update_search`,
                            1, if the rule covers all examples that belong to the difference between the examples that
                            have been provided via the function `Loss#update_sub_sample` and the examples that have been
                            provided via the function `update_search`
        :param accumulated: 0, if the rule covers all examples that have been provided via the function `update_search`
                            since the function `reset_search` has been called for the last time, 1, if the rule covers
                            all examples that have been provided since the search has been started via the the function
                            `Loss#begin_search`
        :return:            A pointer to an object of type `LabelWisePrediction` that stores the scores to be predicted
                            by the rule for each considered label, as well as the corresponding quality scores
        """
        pass

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated):
        """
        Calculates and returns the loss-minimizing scores to be predicted by a rule that covers all examples that have
        been provided to the search so far via the function `update_search`.

        If the argument `uncovered` is 1, the rule is considered to cover all examples that belong to the difference
        between the examples that have been provided via the function `update_sub_sample` and the examples that have
        been provided via the function `update_search`.

        If the argument `accumulated` is 1, all examples that have been provided since the search has been started via
        the function `Loss#begin_search` are taken into account even if the function `reset_search` has been called
        before. If the latter has not been invoked, the argument does not have any effect.

        The calculated scores correspond to the subset of labels that has been provided when starting the search via the
        function `Loss#begin_search`. The score to be predicted for an individual label is calculated with respect to
        the predictions for the other labels. In case of a decomposable loss function, i.e., if the labels are
        considered independently from each other, this function is equivalent to the function
        `calculate_label_wise_prediction`. In addition to the scores, an overall quality score, which assesses the
        quality of the predictions for all labels in terms of a single score, is returned.

        :param uncovered:   0, if the rule covers all examples that have been provided via the function `update_search`,
                            1, if the rule covers all examples that belong to the difference between the examples that
                            have been provided via the function `Loss#update_sub_sample` and the examples that have been
                            provided via the function `update_search`
        :param accumulated: 0, if the rule covers all examples that have been provided via the function `update_search`
                            since the function `reset_search` has been called for the last time, 1, if the rule covers
                            all examples that have been provided since the search has been started via the function
                            `Loss#begin_search`
        :return:            A pointer to an object of type `Prediction` that stores the scores to be predicted by the
                            rule for each considered label, as well as an overall quality score
        """
        pass


cdef class DecomposableRefinementSearch(RefinementSearch):
    """
    A base class for all classes that allow to search for the best refinement of a rule according to a (label-wise)
    decomposable loss.
    """

    cdef void update_search(self, intp example_index, uint32 weight):
        pass

    cdef void reset_search(self):
        pass

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated):
        pass

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated):
        # In case of a decomposable loss, the example-wise predictions are the same as the label-wise predictions...
        return <Prediction*>self.calculate_label_wise_prediction(uncovered, accumulated)


cdef class NonDecomposableRefinementSearch(RefinementSearch):
    """
    A base class for all classes that allow to search for the best refinement of a rule according to a (label-wise)
    non-decomposable loss.
    """

    cdef void update_search(self, intp example_index, uint32 weight):
        pass

    cdef void reset_search(self):
        pass

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated):
        pass

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated):
        pass


cdef class Loss:
    """
    A base class for all (surrogate) loss functions to be minimized locally by the rules learned during training.

    An algorithm for rule induction may use the functions provided by this class to obtain loss-minimizing predictions
    for candidate rules (or the default rule), as well as quality scores that assess the quality of these rules.

    For reasons of efficiency, implementations of this class may be stateful. This enabled to avoid redundant
    recalculations of information that applies to several candidate rules. Call to functions of this class must follow a
    strict protocol regarding the order of function invocations. For detailed information refer to the documentation of
    the individual functions.
    """

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix):
        """
        Calculates the loss-minimizing scores to be predicted by the default rule, i.e., a rule that covers all
        examples, for each label.

        This function must be called exactly once. It must be called prior to the invocation of any other function
        provided by this class.

        As this function is guaranteed to be invoked at first, it may be used to initialize the internal state of an
        instantiation of this class, i.e., to compute and store global information that is required by the other
        functions that will be called later, e.g. overall statistics about the given ground truth labels.

        :param label_matrix:    A `LabelMatrix` that provides random access to the labels of the training examples
        :return:                A pointer to an object of type `DefaultPrediction` that stores the scores to be
                                predicted by the default rule for each label
        """
        pass

    cdef void reset_sampled_examples(self):
        """
        Notifies the loss function that the examples, which should be considered in the following for learning a new
        rule have changed. The indices of the respective examples must be provided via subsequent calls to the function
        `add_sampled_example`.

        This function must be invoked before a new rule is learned from scratch.

        This function is supposed to reset any non-global internal state that only holds for a certain set of examples
        and therefore becomes invalid when different examples are used, e.g. statistics about the ground truth labels of
        particular examples.
        """
        pass

    cdef void add_sampled_example(self, intp example_index, uint32 weight):
        """
        Notifies the loss function about an example that is contained in the sub-sample that should be considered in the
        following for learning a new rule.

        This function must be called repeatedly for each example that should be considered, i.e., for all examples that
        have been selected via instance sub-sampling, immediately after the invocation of the function `reset_examples`.

        This function is supposed to update any internal state that relates to the considered examples, i.e., to compute
        and store local information that is required by the other functions that will be called later, e.g. statistics
        about the ground truth labels of these particular examples. Any information computed by this function is
        expected to be reset when invoking the function `reset_examples` for the next time.

        :param example_index:   The index of an example that should be considered
        :param weight:          The weight of the example that should be considered
        """
        pass

    cdef void reset_covered_examples(self):
        """
        Notifies the loss function that the examples, which should be considered in the following for refining an
        existing rule, have changed. The indices of the respective examples must be provided via subsequent calls to the
        function `update_covered_example`.

        This function must be invoked each time an existing rule has been refined, i.e. when a new condition has been
        added to its body, because this results in fewer examples being covered by the refined rule.

        This function is supposed to reset any non-global internal state that only holds for a certain set of examples
        and therefore becomes invalid when different examples are used, e.g. statistics about the ground truth labels of
        particular examples.
        """
        pass

    cdef void update_covered_example(self, intp example_index, uint32 weight, bint remove):
        """
        Notifies the loss function about an example that is covered by an existing rule and therefore should be
        considered in the following for refining the existing rule.

        This function must be called repeatedly for each example that is covered by the existing rule immediately after
        the invocation of the function `reset_examples`.

        Alternatively, this function may be used to indicate that an example, which has previously been passed to this
        function, should not be considered anymore by setting the argument `remove` accordingly.

        This function is supposed to update any internal state that relates to the considered examples, i.e., to compute
        and store local information that is required by the other functions that will be called later, e.g. statistics
        about the ground truth labels of these particular examples. Any information computed by this function is
        expected to be reset when invoking the function `reset_examples` for the next time.

        :param example_index:   The index of an example that should be considered
        :param weight:          The weight of the example that should be considered
        :param remove:          0, if the example should be considered, 1, if the example should not be considered
                                anymore
        """
        pass

    cdef RefinementSearch begin_search(self, intp[::1] label_indices):
        """
        Starts a new search for the best refinement of a rule. The examples that are covered by such a refinement must
        be provided via subsequent calls to the function `RefinementSearch#update_search`.

        This function must be called each time a new refinement is considered, unless the new refinement covers all
        examples previously provided via calls to the function `RefinementSearch#update_search`.

        Optionally, a subset of the available labels may be specified via the argument `label_indices`. In such case,
        only the specified labels will be considered by the search. When calling this function again to start another
        search from scratch, a different set of labels may be specified.

        :param label_indices:   An array of dtype int, shape `(num_considered_labels)`, representing the indices of the
                                labels that should be considered by the search or None, if all labels should be
                                considered
        :return:                A new object of type `RefinementSearch` to be used to conduct the search
        """
        pass

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, HeadCandidate* head):
        """
        Notifies the loss function about the predictions of a new rule that has been induced.

        This function must be called for each example that is covered by the new rule before learning the next rule,
        i.e., prior to the next invocation of the function `reset_examples`.

        This function is supposed to update any internal state that depends on the predictions of already induced rules.

        :param example_index:       The index of an example that is covered by the newly induced rule, regardless of
                                    whether it is contained in the sub-sample or not
        :param label_indices:       An array of dtype int, shape `(num_predicted_labels)`, representing the indices of
                                    the labels for which the newly induced rule predicts or None, if the rule predicts
                                    for all labels
        :param head:                A pointer to an object of type `HeadCandidate`, representing the head of the newly
                                    induced rule
        """
        pass
