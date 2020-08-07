"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for all classes that allow to store statistics about the labels of training examples.
"""


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

    cdef AbstractRefinementSearch* begin_search(self, intp[::1] label_indices):
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
        :return:                A pointer to an object of type `AbstractRefinementSearch` to be used to conduct the
                                search
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
