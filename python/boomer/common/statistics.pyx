"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for all classes that allow to store statistics about the labels of training examples.
"""


cdef class RefinementSearch:
    """
    A base class for all classes that allow to search for the best refinement of a rule based on previously stored
    statistics.
    """

    cdef void update_search(self, intp statistic_index, uint32 weight):
        """
        Notifies the search that a specific statistic is covered by the condition that is currently considered for
        refining a rule.

        This function must be called repeatedly for each statistic that is covered by the current condition, immediately
        after the invocation of the function `Statistics#begin_search`. Each of these statistics must have been provided
        earlier via the function `Statistics#add_sampled_statistic` or `Statistics#update_covered_statistic`.

        This function is supposed to update any internal state of the search that relates to the examples that are
        covered by the current condition, i.e., to compute and store local information that is required by the other
        functions that will be called later. Any information computed by this function is expected to be reset when
        invoking the function `reset_search` for the next time.

        :param statistic_index: The index of the covered statistic
        :param weight:          The weight of the covered statistic
        """
        pass

    cdef void reset_search(self):
        """
        Resets the internal state of the search that has been updated via preceding calls to the function
        `update_search` to the state when the search was started via the function `Statistics#begin_search`. When
        calling this function, the current state must not be purged entirely, but it must be cached and made available
        for use by the functions `calculate_example_wise_prediction` and `calculate_label_wise_prediction` (if the
        function argument `accumulated` is set accordingly).

        This function may be invoked multiple times with one or several calls to `update_search` in between, which is
        supposed to update the previously cached state by accumulating the current one, i.e., the accumulated cached
        state should be the same as if `reset_search` would not have been called at all.
        """
        pass

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated) nogil:
        """
        Calculates and returns the scores to be predicted by a rule that covers all statistics that have been provided
        to the search so far via the function `update_search`.

        If the argument `uncovered` is 1, the rule is considered to cover all statistics that belong to the difference
        between the statistics that have been provided via the function `add_sampled_statistic` or
        `update_covered_statistic` and the statistics that have been provided via the function `update_search`.

        If the argument `accumulated` is 1, all statistics that have been provided since the search has been started via
        the function `Statistics#begin_search` are taken into account even if the function `reset_search` has been
        called since then. If said function has not been invoked, this argument does not have any effect.

        The calculated scores correspond to the subset of labels that have been provided when starting the search via
        the function `Statistics#begin_search`. The score to be predicted for an individual label is calculated
        independently of the other labels, i.e., in the non-decomposable case it is assumed that the rule will not
        predict for any other labels. In addition to each score, a quality score, which assesses the quality of the
        prediction for the respective label, is returned.

        :param uncovered:   0, if the rule covers all statistics that have been provided via the function
                            `update_search`, 1, if the rule covers all examples that belong to the difference between
                            the statistics that have been provided via the function `Statistics#add_sampled_statistic`
                            or `Statistics#update_covered_statistic` and the statistics that have been provided via the
                            function `update_search`
        :param accumulated: 0, if the rule covers all statistics that have been provided via the function
                            `update_search` since the function `reset_search` has been called for the last time, 1, if
                            the rule covers all examples that have been provided since the search has been started via
                            the function `Statistics#begin_search`
        :return:            A pointer to an object of type `LabelWisePrediction` that stores the scores to be predicted
                            by the rule for each considered label, as well as the corresponding quality scores
        """
        pass

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated) nogil:
        """
        Calculates and returns the scores to be predicted by a rule that covers all statistics that have been provided
        to the search so far via the function `update_search`.

        If the argument `uncovered` is 1, the rule is considered to cover all statistics that belong to the difference
        between the statistics that have been provided via the function `add_sampled_statistic` or
        `update_covered_statistic` and the statistics that have been provided via the function `update_search`.

        If the argument `accumulated` is 1, all statistics that have been provided since the search has been started via
        the function `Statistics#begin_search` are taken into account even if the function `reset_search` has been
        called since then. If said function has not been invoked, this argument does not have any effect.

        The calculated scores correspond to the subset of labels that have been provided when starting the search via
        the function `Statistics#begin_search`. The score to be predicted for an individual label is calculated with
        respect to the predictions for the other labels. In the decomposable case, i.e., if the labels are considered
        independently of each other, this function is equivalent to the function `calculate_label_wise_prediction`. In
        addition to the scores, an overall quality score, which assesses the quality of the predictions for all labels
        in terms of a single score, is returned.

        :param uncovered:   0, if the rule covers all statistics that have been provided via the function
                            `update_search`, 1, if the rule covers all examples that belong to the difference between
                            the statistics that have been provided via the function `Statistics#add_sampled_statistic`
                            or `Statistics#update_covered_statistic` and the statistics that have been provided via the
                            function `update_search`
        :param accumulated: 0, if the rule covers all statistics that have been provided via the function
                            `update_search` since the function `reset_search` has been called for the last time, 1, if
                            the rule covers all examples that have been provided since the search has been started via
                            the function `Statistics#begin_search`
        :return:            A pointer to an object of type `Prediction` that stores the scores to be predicted by the
                            rule for each considered label, as well as an overall quality score
        """
        pass


cdef class Statistics:
    """
    A base class for all classes that store statistics about the labels of the training examples, which serve as the
    basis for learning a new rule or refining an existing one.
    """

    cdef void apply_default_prediction(self, LabelMatrix label_matrix, DefaultPrediction* default_prediction):
        """
        Computes the initial statistics with respect to the predictions of the default rule and the ground truth labels.

        This function must be called exactly once prior to the invocation of any other function provided by this class.

        As this function is guaranteed to be invoked at first, it may be used to initialize any internal state, i.e., to
        compute and store global information that is required by the other function that will be called later.

        :param label_matrix:        A `LabelMatrix` that provides random access to the labels of the training examples
        :param default_prediction:  A pointer to an object of type `DefaultPrediction`, representing the predictions of
                                    the default rule or NULL, if no default rule is available
        """
        pass

    cdef void reset_sampled_statistics(self):
        """
        Resets the statistics which should be considered in the following for learning a new rule. The indices of the
        respective statistics must be provided via subsequent calls to the function `add_sampled_statistic`.

        This function must be invoked before a new rule is learned from scratch, as each rule may be learned on a
        different sub-sample of the statistics.

        This function is supposed to reset any non-global internal state that only holds for a certain subset of the
        available statistics and therefore becomes invalid when a different subset of the statistics should be used.
        """
        pass

    cdef void add_sampled_statistic(self, intp statistic_index, uint32 weight):
        """
        Adds a specific statistic to the sub-sample that should be considered in the following for learning a new rule
        from scratch.

        This function must be called repeatedly for each statistic that should be considered, immediately after the
        invocation of the function `reset_sampled_statistics`.

        This function is supposed to update any internal state that relates to the considered statistics, i.e., to
        compute and store local information that is required by the other function that will be called later. Any
        information computed by this function is expected to be reset when invoking the function
        `reset_sampled_statistics` for the next time.

        :param statistic_index: The index of the statistic that should be considered
        :param weight:          The weight of the statistic that should be considered
        """
        pass

    cdef void reset_covered_statistics(self):
        """
        Resets the statistics which should be considered in the following for refining an existing rule. The indices of
        the respective statistics must be provided via subsequent calls to the function `update_covered_statistic`.

        This function must be invoked each time an existing rule has been refined, i.e., when a new condition has been
        added to its body, because this results in a subset of the statistics being covered by the refined rule.

        This function is supposed to reset any non-global internal state that only holds for a certain subset of the
        available statistics and therefore becomes invalid when a different subset of the statistics should be used.
        """
        pass

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove):
        """
        Adds a specific statistic to the subset that is covered by an existing rule and therefore should be considered
        in the following for refining an existing rule.

        This function must be called repeatedly for each statistic that is covered by the existing rule, immediately
        after the invocation of the function `reset_covered_statistics`.

        Alternatively, this function may be used to indicate that a statistic, which has previously been passed to this
        function, should not be considered anymore by setting the argument `remove` accordingly.

        This function is supposed to update any internal state that relates to the considered statistics, i.e., to
        compute and store local information that is required by the other function that will be called later. Any
        information computed by this function is expected to be reset when invoking the function
        `reset_covered_statistics` for the next time.

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

        This function must be called for each statistic that is covered by the new rule before learning the next rule.

        :param statistic_index: The index of the statistic to be updated
        :param label_indices:   An array of dtype `intp`, shape `(head.numPredictions_)`, representing the indices of
                                the labels for which the newly induced rule predicts or None, if the rule predicts for
                                all labels
        :param head:            A pointer to an object of type `HeadCandidate`, representing the head of the newly
                                induced rule
        """
        pass
