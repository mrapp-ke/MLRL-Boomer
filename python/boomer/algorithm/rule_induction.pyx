# distutils: language=c++

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement algorithms for inducing individual classification rules.
"""
from boomer.algorithm._arrays cimport uint32, float64, array_uint32, array_intp, array_float32, get_index
from boomer.algorithm.rules cimport Head, FullHead, PartialHead, EmptyBody, ConjunctiveBody
from boomer.algorithm.head_refinement cimport HeadCandidate
from boomer.algorithm.losses cimport Prediction

from libc.stdlib cimport qsort

from libcpp.list cimport list
from libcpp.pair cimport pair

from cython.operator cimport dereference, postincrement

from cpython.mem cimport PyMem_Malloc as malloc, PyMem_Realloc as realloc, PyMem_Free as free


cdef class ThresholdProvider:
    """
    A base class for all classes that allow to access the thresholds that can potentially be used by conditions.
    """

    cdef IndexedArray* get_thresholds(self, intp feature_index):
        """
        Creates and returns a pointer to a struct of type `IndexedArray` that stores the indices of training examples,
        as well as their feature values, for a specific feature, sorted in ascending order by the feature values.

        :param feature_index:   The index of the feature
        :return:                A pointer to a struct of type `IndexedArray`
        """
        pass


cdef class DenseThresholdProvider(ThresholdProvider):
    """
    Allows to access the thresholds that can potentially be used by conditions based on the feature values of all
    training examples.

    The feature matrix must be given as a dense Fortran-contiguous array.
    """

    def __cinit__(self, float32[::1, :] x):
        """
        :param x: An array of dtype float, shape `(num_examples, num_features)`, representing the feature values of the
                  training examples
        """
        self.x = x

    cdef IndexedArray* get_thresholds(self, intp feature_index):
        # Class members
        cdef float32[::1, :] x = self.x
        # The number of elements to be returned
        cdef intp num_elements = x.shape[0]
        # The array to be returned
        cdef IndexedValue* sorted_array = <IndexedValue*>malloc(num_elements * sizeof(IndexedValue))
        # The struct to be returned
        cdef IndexedArray* indexed_array = <IndexedArray*>malloc(sizeof(IndexedArray))
        dereference(indexed_array).num_elements = num_elements
        dereference(indexed_array).data = sorted_array
        # Temporary variables
        cdef intp i

        for i in range(num_elements):
            sorted_array[i].index = i
            sorted_array[i].value = x[i, feature_index]

        qsort(sorted_array, num_elements, sizeof(IndexedValue), &__compare_indexed_value)
        return indexed_array


cdef class SparseThresholdProvider(ThresholdProvider):
    """
    Allows to access the thresholds that can potentially be used by conditions based on the feature values of all
    training examples.

    The feature matrix must be given in compressed sparse column (CSC) format.
    """

    def __cinit__(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices):
        """
        :param x_data:          An array of dtype float, shape `(num_non_zero_feature_values)`, representing the
                                non-zero feature values of the training examples
        :param x_row_indices:   An array of dtype int, shape `(num_non_zero_feature_values)`, representing the
                                row-indices of the examples, the values in `x_data` correspond to
        :param x_col_indices:   An array of dtype int, shape `(num_features + 1)`, representing the indices of the first
                                element in `x_data` and `x_row_indices` that corresponds to a certain feature. The index
                                at the last position is equal to `num_non_zero_feature_values`
        """
        self.x_data = x_data
        self.x_row_indices = x_row_indices
        self.x_col_indices = x_col_indices

    cdef IndexedArray* get_thresholds(self, intp feature_index):
        # Class members
        cdef float32[::1] x_data = self.x_data
        cdef intp[::1] x_row_indices = self.x_row_indices
        cdef intp[::1] x_col_indices = self.x_col_indices
        # The index of the first element in `x_data` and `x_row_indices` that corresponds to the given feature index
        cdef intp start = x_col_indices[feature_index]
        # The index of the last element in `x_data` and `x_row_indices` that corresponds to the given feature index
        cdef intp end = x_col_indices[feature_index + 1]
        # The number of elements to be returned
        cdef intp num_elements = end - start
        # The struct to be returned
        cdef IndexedArray* indexed_array = <IndexedArray*>malloc(sizeof(IndexedArray))
        dereference(indexed_array).num_elements = num_elements
        # The array to be returned
        cdef IndexedValue* sorted_array = NULL
        # Temporary variables
        cdef intp i, j

        if num_elements > 0:
            sorted_array = <IndexedValue*>malloc(num_elements * sizeof(IndexedValue))
            i = 0

            for j in range(start, end):
                sorted_array[i].index = x_row_indices[j]
                sorted_array[i].value = x_data[j]
                i += 1

            qsort(sorted_array, num_elements, sizeof(IndexedValue), &__compare_indexed_value)

        dereference(indexed_array).data = sorted_array
        return indexed_array


cdef class RuleInduction:
    """
    A base class for all classes that implement an algorithm for the induction of individual classification rules.
    """

    cdef Rule induce_default_rule(self, uint8[::1, :] y, Loss loss):
        """
        Induces the default rule that minimizes a certain loss function with respect to the given ground truth labels.

        :param y:       An array of dtype float, shape `(num_examples, num_labels)`, representing the ground truth
                        labels of the training examples
        :param loss:    The loss function to be minimized
        :return:        The default rule that has been induced
        """
        pass

    cdef Rule induce_rule(self, intp[::1] nominal_attribute_indices, ThresholdProvider threshold_provider,
                          intp num_examples, intp num_features, intp num_labels, HeadRefinement head_refinement,
                          Loss loss, LabelSubSampling label_sub_sampling, InstanceSubSampling instance_sub_sampling,
                          FeatureSubSampling feature_sub_sampling, Pruning pruning, Shrinkage shrinkage,
                          intp min_coverage, intp max_conditions, RNG rng):
        """
        Induces a single- or multi-label classification rule that minimizes a certain loss function for the training
        examples it covers.

        :param nominal_attribute_indices:   An array of dtype int, shape `(num_nominal_attributes)`, representing the
                                            indices of all nominal features (in ascending order) or None, if no nominal
                                            features are available
        :param threshold_provider:          A `ThresholdProvider` that allows to access the thresholds that can
                                            potentially be used by conditions
        :param num_examples:                The total number of training examples
        :param num_features:                The total number of features
        :param num_labels:                  The total number of labels
        :param head_refinement:             The strategy that is used to find the heads of rules
        :param loss:                        The loss function to be minimized
        :param label_sub_sampling:          The strategy that should be used to sub-sample the labels or None, if no
                                            label sub-sampling should be used
        :param instance_sub_sampling:       The strategy that should be used to sub-sample the training examples or
                                            None, if no instance sub-sampling should be used
        :param feature_sub_sampling:        The strategy that should be used to sub-sample the available features or
                                            None, if no feature sub-sampling should be used
        :param pruning:                     The strategy that should be used to prune rules or None, if no pruning
                                            should be used
        :param shrinkage:                   The strategy that should be used to shrink the weights of rules or None, if
                                            no shrinkage should be used
        :param min_coverage:                The minimum number of training examples that must be covered by the rule.
                                            Must be at least 1
        :param max_conditions:              The maximum number of conditions to be included in the rule's body. Must be
                                            at least 1 or -1, if the number of conditions should not be restricted
        :param rng:                         The random number generator to be used
        :return:                            The rule that has been induced or None, if no rule could be induced
        """
        pass


cdef class ExactGreedyRuleInduction(RuleInduction):
    """
    Allows to induce single- or multi-label classification rules using a greedy search, where new conditions are added
    iteratively to the (initially empty) body of a rule. At each iteration, the refinement that improves the rule the
    most is chosen. The search stops if no refinement results in an improvement. The possible conditions to be evaluated
    at each iteration result from an exact split finding algorithm, i.e., all possible thresholds that may be used by
    the conditions are considered.
    """

    def __cinit__(self):
        self.cache_global = new map[intp, IndexedArray*]()

    def __dealloc__(self):
        cdef map[intp, IndexedArray*]* cache_global = self.cache_global
        cdef map[intp, IndexedArray*].iterator cache_global_iterator = dereference(cache_global).begin()
        cdef IndexedArray* indexed_array

        while cache_global_iterator != dereference(cache_global).end():
            indexed_array = dereference(cache_global_iterator).second
            free(dereference(indexed_array).data)
            free(indexed_array)
            postincrement(cache_global_iterator)

        del self.cache_global

    cdef Rule induce_default_rule(self, uint8[::1, :] y, Loss loss):
        cdef float64[::1] scores = loss.calculate_default_scores(y)
        cdef FullHead head = FullHead.__new__(FullHead, scores)
        cdef EmptyBody body = EmptyBody.__new__(EmptyBody)
        cdef Rule rule = Rule.__new__(Rule, body, head)
        return rule

    cdef Rule induce_rule(self, intp[::1] nominal_attribute_indices, ThresholdProvider threshold_provider,
                          intp num_examples, intp num_features, intp num_labels, HeadRefinement head_refinement,
                          Loss loss, LabelSubSampling label_sub_sampling, InstanceSubSampling instance_sub_sampling,
                          FeatureSubSampling feature_sub_sampling, Pruning pruning, Shrinkage shrinkage,
                          intp min_coverage, intp max_conditions, RNG rng):
        # The head of the induced rule
        cdef HeadCandidate head = None
        # A (stack-allocated) list that contains the conditions in the rule's body (in the order they have been learned)
        cdef list[Condition] conditions
        # The total number of conditions
        cdef intp num_conditions = 0
        # An array representing the number of conditions per type of operator
        cdef intp[::1] num_conditions_per_comparator = array_intp(4)
        num_conditions_per_comparator[:] = 0
        # An array that is used to keep track of the indices of the training examples are covered by the current rule.
        # Each element in the array corresponds to the example at the corresponding index. If the value for an element
        # is equal to `covered_examples_target`, it is covered by the current rule, otherwise it is not.
        cdef uint32[::1] covered_examples_mask = array_uint32(num_examples)
        covered_examples_mask[:] = 0
        cdef uint32 covered_examples_target = 0

        # Variables for representing the best refinement
        cdef bint found_refinement = True
        cdef Comparator best_condition_comparator
        cdef intp best_condition_start, best_condition_end, best_condition_previous, best_condition_feature_index
        cdef float32 best_condition_threshold
        cdef intp best_condition_covered_weights, best_condition_num_indexed_values
        cdef IndexedValue* best_condition_indexed_values
        cdef IndexedArrayWrapper* best_condition_indexed_array_wrapper

        # Variables for specifying the examples that should be used for finding the best refinement
        cdef map[intp, IndexedArray*]* cache_global = self.cache_global
        cdef IndexedArray* indexed_array
        cdef map[intp, IndexedArrayWrapper*] cache_local  # Stack-allocated map
        cdef map[intp, IndexedArrayWrapper*].iterator cache_local_iterator
        cdef IndexedArrayWrapper* indexed_array_wrapper
        cdef IndexedValue* indexed_values
        cdef intp num_indexed_values
        cdef bint sparse

        # Variables for specifying the features that should be used for finding the best refinement
        cdef intp num_nominal_features = nominal_attribute_indices.shape[0] if nominal_attribute_indices is not None else 0
        cdef intp next_nominal_f = -1
        cdef intp[::1] feature_indices
        cdef intp next_nominal_c, num_sampled_features
        cdef bint nominal

        # Temporary variables
        cdef HeadCandidate current_head
        cdef Prediction prediction
        cdef float64[::1] predicted_scores
        cdef float32 previous_threshold, current_threshold
        cdef uint32 weight
        cdef intp c, f, r, i, first_r, previous_r

        # Sub-sample examples, if necessary...
        cdef pair[uint32[::1], intp] instance_sub_sampling_result
        cdef uint32[::1] weights
        cdef intp total_sum_of_weights, sum_of_weights, accumulated_sum_of_weights

        if instance_sub_sampling is None:
            weights = None
            total_sum_of_weights = num_examples
        else:
            instance_sub_sampling_result = instance_sub_sampling.sub_sample(num_examples, rng)
            weights = instance_sub_sampling_result.first
            total_sum_of_weights = instance_sub_sampling_result.second

        # Notify the loss function about the examples that are included in the sub-sample...
        loss.begin_instance_sub_sampling()

        for i in range(num_examples):
            weight = 1 if weights is None else weights[i]
            loss.update_sub_sample(i, weight)

        # Sub-sample labels, if necessary...
        cdef intp[::1] label_indices

        if label_sub_sampling is None:
            label_indices = None
        else:
            label_indices = label_sub_sampling.sub_sample(num_labels, rng)

        try:
            # Search for the best refinement until no improvement in terms of the rule's quality score is possible
            # anymore or the maximum number of conditions has been reached...
            while found_refinement and (max_conditions == -1 or num_conditions < max_conditions):
                found_refinement = False

                # Sub-sample features, if necessary...
                if feature_sub_sampling is None:
                    feature_indices = None
                    num_sampled_features = num_features
                else:
                    feature_indices = feature_sub_sampling.sub_sample(num_features, rng)
                    num_sampled_features = feature_indices.shape[0]

                # Obtain the index of the first nominal feature, if any...
                if num_nominal_features > 0:
                    next_nominal_f = nominal_attribute_indices[0]
                    next_nominal_c = 1

                # Search for the best condition among all available features to be added to the current rule. For each
                # feature, the examples are traversed in descending order of their respective feature values and the
                # loss function is updated accordingly. For each potential condition, a quality score is calculated to
                # keep track of the best possible refinement.
                for c in range(num_sampled_features):
                    f = get_index(c, feature_indices)

                    # Obtain array that contains the indices of the training examples sorted according to the current
                    # feature...
                    indexed_array_wrapper = cache_local[f]

                    if indexed_array_wrapper == NULL:
                        indexed_array_wrapper = <IndexedArrayWrapper*>malloc(sizeof(IndexedArrayWrapper))
                        dereference(indexed_array_wrapper).array = NULL
                        dereference(indexed_array_wrapper).num_conditions = 0
                        cache_local[f] = indexed_array_wrapper

                    indexed_array = dereference(indexed_array_wrapper).array

                    if indexed_array == NULL:
                        indexed_array = dereference(cache_global)[f]

                        if indexed_array == NULL:
                            indexed_array = threshold_provider.get_thresholds(f)
                            dereference(cache_global)[f] = indexed_array

                    # Filter indices, if only a subset of the contained examples is covered...
                    if num_conditions > dereference(indexed_array_wrapper).num_conditions:
                        __filter_any_indices(indexed_array, indexed_array_wrapper, num_conditions,
                                             covered_examples_mask, covered_examples_target)
                        indexed_array = dereference(indexed_array_wrapper).array

                    num_indexed_values = dereference(indexed_array).num_elements
                    indexed_values = dereference(indexed_array).data

                    # Check if feature is nominal...
                    if f == next_nominal_f:
                        nominal = True

                        if next_nominal_c < num_nominal_features:
                            next_nominal_f = nominal_attribute_indices[next_nominal_c]
                            next_nominal_c += 1
                        else:
                            next_nominal_f = -1
                    else:
                        nominal = False

                    # Reset the loss function when processing a new feature...
                    loss.begin_search(label_indices)
                    sum_of_weights = 0
                    accumulated_sum_of_weights = 0
                    first_r = num_indexed_values - 1

                    # Traverse examples in descending order until the first example with weight > 0 is encountered...
                    for r in range(first_r, -1, -1):
                        i = indexed_values[r].index
                        weight = 1 if weights is None else weights[i]

                        if weight > 0:
                            # Tell the loss function that the example will be covered by upcoming refinements...
                            loss.update_search(i, weight)
                            sum_of_weights += weight
                            accumulated_sum_of_weights += weight
                            previous_threshold = indexed_values[r].value
                            previous_r = r
                            break

                    # Traverse the remaining examples in descending order...
                    for r in range(r - 1, -1, -1):
                        i = indexed_values[r].index
                        weight = 1 if weights is None else weights[i]

                        # Do only consider examples that are included in the current sub-sample...
                        if weight > 0:
                            current_threshold = indexed_values[r].value

                            # Split points between examples with the same feature value must not be considered...
                            if previous_threshold != current_threshold:
                                # Find and evaluate the best head for the current refinement, if a condition that uses
                                # the > operator (or the == operator in case of a nominal feature) is used...
                                current_head = head_refinement.find_head(head, label_indices, loss, False, False)

                                # If refinement using the > operator (or the == operator in case of a nominal feature)
                                # is better than the current rule...
                                if current_head is not None:
                                    found_refinement = True
                                    head = current_head
                                    best_condition_start = first_r
                                    best_condition_end = r
                                    best_condition_previous = previous_r
                                    best_condition_feature_index = f
                                    best_condition_covered_weights = sum_of_weights
                                    best_condition_num_indexed_values = num_indexed_values
                                    best_condition_indexed_values = indexed_values
                                    best_condition_indexed_array_wrapper = indexed_array_wrapper

                                    if nominal:
                                        best_condition_comparator = Comparator.EQ
                                        best_condition_threshold = previous_threshold
                                    else:
                                        best_condition_comparator = Comparator.GR
                                        best_condition_threshold = (previous_threshold + current_threshold) / 2.0

                                # Find and evaluate the best head for the current refinement, if a condition that uses
                                # the <= operator (or the != operator in case of a nominal feature) is used...
                                current_head = head_refinement.find_head(head, label_indices, loss, True, False)

                                # If refinement using the <= operator (or the != operator in case of a nominal feature)
                                # is better than the current rule...
                                if current_head is not None:
                                    found_refinement = True
                                    head = current_head
                                    best_condition_start = first_r
                                    best_condition_end = r
                                    best_condition_previous = previous_r
                                    best_condition_feature_index = f
                                    best_condition_covered_weights = (total_sum_of_weights - sum_of_weights)
                                    best_condition_num_indexed_values = num_indexed_values
                                    best_condition_indexed_values = indexed_values
                                    best_condition_indexed_array_wrapper = indexed_array_wrapper

                                    if nominal:
                                        best_condition_comparator = Comparator.NEQ
                                        best_condition_threshold = previous_threshold
                                    else:
                                        best_condition_comparator = Comparator.LEQ
                                        best_condition_threshold = (previous_threshold + current_threshold) / 2.0

                                # Reset the loss function in case of a nominal feature, as the previous examples will
                                # not be covered by the next condition...
                                if nominal:
                                    loss.reset_search()
                                    sum_of_weights = 0
                                    first_r = r

                            previous_threshold = current_threshold
                            previous_r = r

                            # Tell the loss function that the example will be covered by upcoming refinements...
                            loss.update_search(i, weight)
                            sum_of_weights += weight
                            accumulated_sum_of_weights += weight

                    # If not all examples have been iterated, this means that there are examples with (sparse) feature
                    # value == 0. In such case, we must explicitly test conditions that separate these examples from the
                    # ones that have already been iterated...
                    sparse = accumulated_sum_of_weights < total_sum_of_weights

                    if sparse:
                        # Find and evaluate the best head for the current refinement, if a condition that uses the >
                        # operator (or the == operator in case of a nominal feature) is used...
                        current_head = head_refinement.find_head(head, label_indices, loss, False, False)

                        # If refinement using the > operator (or the == operator in case of a nominal feature) is better
                        # than the current rule...
                        if current_head is not None:
                            found_refinement = True
                            head = current_head
                            best_condition_start = first_r
                            best_condition_end = -1
                            best_condition_previous = previous_r
                            best_condition_feature_index = f
                            best_condition_covered_weights = sum_of_weights
                            best_condition_num_indexed_values = num_indexed_values
                            best_condition_indexed_values = indexed_values
                            best_condition_indexed_array_wrapper = indexed_array_wrapper

                            if nominal:
                                best_condition_comparator = Comparator.EQ
                                best_condition_threshold = previous_threshold
                            else:
                                best_condition_comparator = Comparator.GR
                                best_condition_threshold = previous_threshold / 2.0

                        # Find and evaluate the best head for the current refinement, if a condition that uses the <=
                        # operator (or the != operator in case of a nominal feature) is used...
                        current_head = head_refinement.find_head(head, label_indices, loss, True, False)

                        # If refinement using the <= operator (or the != operator in case of a nominal feature) is
                        # better than the current rule...
                        if current_head is not None:
                            found_refinement = True
                            head = current_head
                            best_condition_start = first_r
                            best_condition_end = -1
                            best_condition_previous = previous_r
                            best_condition_feature_index = f
                            best_condition_covered_weights = (total_sum_of_weights - sum_of_weights)
                            best_condition_num_indexed_values = num_indexed_values
                            best_condition_indexed_values = indexed_values
                            best_condition_indexed_array_wrapper = indexed_array_wrapper

                            if nominal:
                                best_condition_comparator = Comparator.NEQ
                                best_condition_threshold = previous_threshold
                            else:
                                best_condition_comparator = Comparator.LEQ
                                best_condition_threshold = previous_threshold / 2.0

                    # If the feature is nominal and there are examples with different feature values, we must evaluate
                    # additional conditions...
                    if nominal and (sparse or sum_of_weights < total_sum_of_weights):
                        # Find and evaluate the best head for the current refinement, if a condition that uses ==
                        # operator is used...
                        current_head = head_refinement.find_head(head, label_indices, loss, sparse, sparse)

                        # If refinement using the == operator is better than the current rule...
                        if current_head is not None:
                            found_refinement = True
                            head = current_head
                            best_condition_start = first_r
                            best_condition_end = -1
                            best_condition_previous = previous_r
                            best_condition_feature_index = f

                            if sparse:
                                best_condition_covered_weights = (total_sum_of_weights - accumulated_sum_of_weights)
                                best_condition_threshold = 0.0
                            else:
                                best_condition_covered_weights = sum_of_weights
                                best_condition_threshold = previous_threshold

                            best_condition_num_indexed_values = num_indexed_values
                            best_condition_indexed_values = indexed_values
                            best_condition_indexed_array_wrapper = indexed_array_wrapper
                            best_condition_comparator = Comparator.EQ

                        # Find and evaluate the best head for the current refinement, if a condition that uses the !=
                        # operator is used...
                        current_head = head_refinement.find_head(head, label_indices, loss, not sparse, sparse)

                        # If refinement using the != operator is better than the current rule...
                        if current_head is not None:
                            found_refinement = True
                            head = current_head
                            best_condition_start = first_r
                            best_condition_end = -1
                            best_condition_previous = previous_r
                            best_condition_feature_index = f

                            if sparse:
                                best_condition_covered_weights = accumulated_sum_of_weights
                                best_condition_threshold = 0.0
                            else:
                                best_condition_covered_weights = (total_sum_of_weights - sum_of_weights)
                                best_condition_threshold = previous_threshold

                            best_condition_num_indexed_values = num_indexed_values
                            best_condition_indexed_values = indexed_values
                            best_condition_indexed_array_wrapper = indexed_array_wrapper
                            best_condition_comparator = Comparator.NEQ

                if found_refinement:
                    # If a refinement has been found, add the new condition and update the labels for which the rule
                    # predicts...
                    conditions.push_back(__make_condition(best_condition_feature_index, best_condition_comparator,
                                                          best_condition_threshold))
                    num_conditions += 1
                    num_conditions_per_comparator[<intp>best_condition_comparator] += 1
                    label_indices = head.label_indices

                    # If instance sub-sampling is used, examples that are not contained in the current sub-sample were
                    # not considered for finding the new condition. In the next step, we need to identify the examples
                    # that are covered by the refined rule, including those that are not contained in the sub-sample,
                    # via the function `__filter_current_indices`. Said function calculates the number of covered
                    # examples based on the variable `best_condition_end`, which represents the position that separates
                    # the covered from the uncovered examples. However, when taking into account the examples that are
                    # not contained in the sub-sample, this position may differ from the current value of
                    # `best_condition_end` and therefore must be adjusted...
                    if weights is not None and best_condition_previous - best_condition_end > 1:
                        best_condition_end = __adjust_split(best_condition_indexed_values, best_condition_end,
                                                            best_condition_previous, best_condition_threshold)

                    # Identify the examples for which the rule predicts...
                    covered_examples_target = __filter_current_indices(best_condition_indexed_values,
                                                                       best_condition_num_indexed_values,
                                                                       best_condition_indexed_array_wrapper,
                                                                       best_condition_start, best_condition_end,
                                                                       best_condition_comparator, num_conditions,
                                                                       covered_examples_mask, covered_examples_target,
                                                                       loss, weights)
                    total_sum_of_weights = best_condition_covered_weights

                    if total_sum_of_weights <= min_coverage:
                        # Abort refinement process if rule is not allowed to cover less examples...
                        break

            if head is None:
                # No rule could be induced, because no useful condition could be found. This is for example the case, if
                # all features are constant.
                return None
            else:
                predicted_scores = head.predicted_scores

                if weights is not None:
                    # Prune rule, if necessary (a rule can only be pruned if it contains more than one condition)...
                    if pruning is not None and num_conditions > 1:
                        # TODO revise pruning
                        #pruning.begin_pruning(weights, loss, head_refinement, covered_example_indices, label_indices)
                        #covered_example_indices = pruning.prune(x, cache_global, conditions)
                        #num_covered = covered_example_indices.shape[0]
                        print('pruning not supported right now')

                    # If instance sub-sampling is used, we need to re-calculate the scores in the head based on the
                    # entire training data...
                    loss.begin_search(label_indices)

                    for r in range(num_examples):
                        if covered_examples_mask[r] == covered_examples_target:
                            loss.update_search(i, 1)

                    prediction = head_refinement.evaluate_predictions(loss, False, False)
                    predicted_scores[:] = prediction.predicted_scores

                # Apply shrinkage, if necessary...
                if shrinkage is not None:
                    shrinkage.apply_shrinkage(predicted_scores)

                # Tell the loss function that a new rule has been induced...
                for r in range(num_examples):
                    if covered_examples_mask[r] == covered_examples_target:
                        loss.apply_prediction(r, label_indices, predicted_scores)

                # Build and return the induced rule...
                return __build_rule(label_indices, predicted_scores, conditions, num_conditions_per_comparator)
        finally:
            # Free memory occupied by the arrays stored in `cache_local`...
            cache_local_iterator = cache_local.begin()

            while cache_local_iterator != cache_local.end():
                indexed_array_wrapper = dereference(cache_local_iterator).second
                indexed_array = dereference(indexed_array_wrapper).array

                if indexed_array != NULL:
                    indexed_values = dereference(indexed_array).data
                    free(indexed_values)

                free(indexed_array)
                free(indexed_array_wrapper)
                postincrement(cache_local_iterator)


cdef int __compare_indexed_value(const void* a, const void* b) nogil:
    """
    Compares the values of two structs of type `IndexedValue`.

    :param a:   A pointer to the first struct
    :param b:   A pointer to the second struct
    :return:    -1 if the value of the first struct is smaller than the value of the second struct, 0 if both values are
                equal, or 1 if the value of the first struct is greater than the value of the second struct
    """
    cdef float32 v1 = (<IndexedValue*>a).value
    cdef float32 v2 = (<IndexedValue*>b).value
    return -1 if v1 < v2 else (0 if v1 == v2 else 1)


cdef inline Condition __make_condition(intp feature_index, Comparator comparator, float32 threshold):
    """
    Creates and returns a new condition.

    :param feature_index:   The index of the feature that is used by the condition
    :param comparator:      The type of the operator used by the condition
    :param threshold:       The threshold that is used by the condition
    """
    cdef Condition condition
    condition.feature_index = feature_index
    condition.comparator = comparator
    condition.threshold = threshold
    return condition


cdef inline intp __adjust_split(IndexedValue* indexed_values, intp position_start, intp position_end,
                                float32 threshold):
    """
    Adjusts the position that separates the covered from the uncovered examples with respect to those examples that are
    not contained in the current sub-sample. This requires to look back a certain number of examples, i.e., to traverse
    the examples in ascending order until the next example that is contained in the current sub-sample is encountered,
    to see if they satisfy the new condition or not.

    :param indexed_values:  A pointer to a C-array of type `IndexedValue` that stores the indices of the training
                            examples, as well as the corresponding feature values, sorted in ascending order according
                            to the feature values
    :param position_start:  The position that separates the covered from the uncovered examples (when only taking into
                            account the examples that are contained in the sample). This is the position to start at
    :param position_end:    The position to stop at (exclusive, must be greater than `position_start`)
    :param threshold:       The threshold of the condition
    :return:                The adjusted position that separates the covered from the uncovered examples with respect to
                            the examples that are not contained in the sample
    """
    cdef intp adjusted_position = position_start
    cdef float32 feature_value
    cdef intp r

    # Traverse the examples in ascending order until we encounter an example that is contained in the current
    # sub-sample...
    for r in range(position_start + 1, position_end):
        feature_value = indexed_values[r].value

        if feature_value <= threshold:
            # The feature value at `position_start` is guaranteed to be smaller than or equal to the given `threshold`.
            # If this does also apply to the feature value of a preceding example, it is not separated from the example
            # at `position_start`. Hence, we are not done yet and continue by updating the adjusted position...
            adjusted_position = r
        else:
            # If we have found the first example that is separated from the example at the position we started at, we
            # are done...
            break

    return adjusted_position


cdef inline uint32 __filter_current_indices(IndexedValue* indexed_values, intp num_indexed_values,
                                            IndexedArrayWrapper* indexed_array_wrapper, intp condition_start,
                                            intp condition_end, Comparator condition_comparator, intp num_conditions,
                                            uint32[::1] covered_examples_mask, uint32 covered_examples_target,
                                            Loss loss, uint32[::1] weights):
    """
    Filters an array that contains the indices of the examples that are covered by the previous rule, as well as their
    values for a certain feature, after a new condition that corresponds to said feature has been added, such that the
    filtered array does only contain the indices and feature values of the examples that are covered by the new rule.
    The filtered array is stored in a given struct of type `IndexedArrayWrapper`.

    :param indexed_values:          A pointer to a C-array of type `IndexedValue` that stores the indices of the
                                    training examples that are covered by the previous rule, as well as their feature
                                    values for the feature, the new condition corresponds to, sorted in ascending order
                                    according to the feature values
    :param num_indexed_values:      The number of elements in the array `indexed_values`
    :param indexed_array_wrapper:   A pointer to a struct of type `IndexedArrayWrapper` that should be used to store the
                                    filtered array
    :param condition_start:         The element in `indexed_values` that corresponds to the first example (inclusive)
                                    that has been passed to the loss function when searching for the new condition (must
                                    be greater than `condition_end`)
    :param condition_end:           The element in `indexed_values` that corresponds to the last example (exclusive)
                                    that has been passed to the loss function when searching for the new condition (must
                                    be smaller than `condition_start`)
    :param condition_comparator:    The type of the operator that is used by the new condition
    :param num_conditions:          The total number of conditions in the rule's body (including the new one)
    :param covered_examples_mask:   An array of dtype uint, shape `(num_examples)` that is used to keep track of the
                                    indices of the examples that are covered by the previous rule. It will be updated by
                                    this function
    :param covered_examples_target: The value that is used to mark those elements in `covered_examples_mask` that are
                                    covered by the previous rule
    :param loss:                    The loss function to be notified about the examples that must be considered when
                                    searching for the next refinement, i.e., the examples that are covered by the new
                                    rule
    :param weights:                 An array of dtype uint, shape `(num_examples)`, representing the weights of the
                                    training examples
    :return:                        The value that is used to mark those elements in the updated `covered_examples_mask`
                                    that are covered by the new rule
    """
    cdef intp num_elements = condition_start - condition_end

    if condition_comparator == Comparator.LEQ or condition_comparator == Comparator.NEQ:
        num_elements = num_indexed_values - num_elements

    cdef IndexedValue* filtered_array = <IndexedValue*>malloc(num_elements * sizeof(IndexedValue))
    cdef intp i = num_elements - 1
    cdef uint32 updated_target, weight
    cdef intp r, index

    if condition_comparator == Comparator.GR or condition_comparator == Comparator.EQ:
        updated_target = num_conditions
        loss.begin_instance_sub_sampling()

        for r in range(condition_start, condition_end, -1):
            index = indexed_values[r].index
            covered_examples_mask[index] = num_conditions
            filtered_array[i].index = index
            filtered_array[i].value = indexed_values[r].value
            weight = 1 if weights is None else weights[index]
            loss.update_sub_sample(index, weight)
            i -= 1
    else:
        updated_target = covered_examples_target

        if condition_comparator == Comparator.NEQ:
            for r in range(num_indexed_values - 1, condition_start, -1):
                filtered_array[i].index = indexed_values[r].index
                filtered_array[i].value = indexed_values[r].value
                i -= 1

        for r in range(condition_start, condition_end, -1):
            index = indexed_values[r].index
            covered_examples_mask[index] = num_conditions
            weight = 1 if weights is None else weights[index]
            loss.remove_from_sub_sample(index, weight)

        for r in range(condition_end, -1, -1):
            filtered_array[i].index = indexed_values[r].index
            filtered_array[i].value = indexed_values[r].value
            i -= 1

    cdef IndexedArray* indexed_array = dereference(indexed_array_wrapper).array

    if indexed_array == NULL:
        indexed_array = <IndexedArray*>malloc(sizeof(IndexedArray))
        dereference(indexed_array_wrapper).array = indexed_array
    else:
        free(dereference(indexed_array).data)

    dereference(indexed_array).data = filtered_array
    dereference(indexed_array).num_elements = num_elements
    dereference(indexed_array_wrapper).num_conditions = num_conditions
    return updated_target


cdef inline void __filter_any_indices(IndexedArray* indexed_array, IndexedArrayWrapper* indexed_array_wrapper,
                                      intp num_conditions, uint32[::1] covered_examples_mask,
                                      uint32 covered_examples_target):
    """
    Filters an array that contains the indices of examples, as well as their values for a certain feature, such that the
    filtered array does only contain the indices and feature values of the examples that are covered by the current
    rule. The filtered array is stored in a given struct of type `IndexedArrayWrapper`.

    :param indexed_array:           A pointer to a struct of type `IndexedArray` that stores a pointer to the C-array to
                                    be filtered, as well as the number of elements in said array
    :param indexed_array_wrapper:   A pointer to a struct of type `IndexedArrayWrapper` that should be used to store the
                                    filtered array
    :param num_conditions:          The total number of conditions in the current rule's body
    :param covered_examples_mask:   An array of dtype uint, shape `(num_examples)` that is used to keep track of the
                                    indices of the examples that are covered by the previous rule. It will be updated by
                                    this function
    :param covered_examples_target: The value that is used to mark those elements in `covered_examples_mask` that are
                                    covered by the previous rule
    """
    cdef IndexedArray* filtered_indexed_array = dereference(indexed_array_wrapper).array
    cdef IndexedValue* filtered_array = NULL

    if filtered_indexed_array != NULL:
        filtered_array = dereference(filtered_indexed_array).data

    cdef intp max_elements = dereference(indexed_array).num_elements
    cdef intp i = 0
    cdef IndexedValue* indexed_values
    cdef intp r, index

    if max_elements > 0:
        indexed_values = dereference(indexed_array).data

        if filtered_array == NULL:
            filtered_array = <IndexedValue*>malloc(max_elements * sizeof(IndexedValue))

        for r in range(max_elements):
            index = indexed_values[r].index

            if covered_examples_mask[index] == covered_examples_target:
                filtered_array[i].index = index
                filtered_array[i].value = indexed_values[r].value
                i += 1

    if i == 0:
        free(filtered_array)
        filtered_array = NULL
    elif i < max_elements:
        filtered_array = <IndexedValue*>realloc(filtered_array, i * sizeof(IndexedValue))

    if filtered_indexed_array == NULL:
        filtered_indexed_array = <IndexedArray*>malloc(sizeof(IndexedArray))

    dereference(filtered_indexed_array).data = filtered_array
    dereference(filtered_indexed_array).num_elements = i
    dereference(indexed_array_wrapper).array = filtered_indexed_array
    dereference(indexed_array_wrapper).num_conditions = num_conditions


cdef inline Rule __build_rule(intp[::1] label_indices, float64[::1] predicted_scores, list[Condition] conditions,
                              intp[::1] num_conditions_per_comparator):
    """
    Builds and returns a rule.

    :param label_indices:                   An array of dtype int, shape `(num_predicted_labels)`, representing the
                                            indices of the labels for which the rule predicts or None, if the rule
                                            predicts for all labels
    :param predicted_scores:                An array of dtype float, shape `(num_predicted_labels)`, representing the
                                            scores that are predicted by the rule
    :param conditions:                      A list that contains the rule's conditions
    :param num_conditions_per_comparator:   An array of dtype int, shape `(4)`, representing the number of conditions
                                            that use a specific operator
    return:                                 The rule that has been built
    """
    cdef intp num_conditions = num_conditions_per_comparator[<intp>Comparator.LEQ]
    cdef intp[::1] leq_feature_indices = array_intp(num_conditions) if num_conditions > 0 else None
    cdef float32[::1] leq_thresholds = array_float32(num_conditions) if num_conditions > 0 else None
    num_conditions = num_conditions_per_comparator[<intp>Comparator.GR]
    cdef intp[::1] gr_feature_indices = array_intp(num_conditions) if num_conditions > 0 else None
    cdef float32[::1] gr_thresholds = array_float32(num_conditions) if num_conditions > 0 else None
    num_conditions = num_conditions_per_comparator[<intp>Comparator.EQ]
    cdef intp[::1] eq_feature_indices = array_intp(num_conditions) if num_conditions > 0 else None
    cdef float32[::1] eq_thresholds = array_float32(num_conditions) if num_conditions > 0 else None
    num_conditions = num_conditions_per_comparator[<intp>Comparator.NEQ]
    cdef intp[::1] neq_feature_indices = array_intp(num_conditions) if num_conditions > 0 else None
    cdef float32[::1] neq_thresholds = array_float32(num_conditions) if num_conditions > 0 else None
    cdef list[Condition].iterator iterator = conditions.begin()
    cdef intp leq_i = 0
    cdef intp gr_i = 0
    cdef intp eq_i = 0
    cdef intp neq_i = 0
    cdef Condition condition
    cdef Comparator comparator

    while iterator != conditions.end():
        condition = dereference(iterator)
        comparator = condition.comparator

        if comparator == Comparator.LEQ:
           leq_feature_indices[leq_i] = condition.feature_index
           leq_thresholds[leq_i] = condition.threshold
           leq_i += 1
        elif comparator == Comparator.GR:
           gr_feature_indices[gr_i] = condition.feature_index
           gr_thresholds[gr_i] = condition.threshold
           gr_i += 1
        elif comparator == Comparator.EQ:
           eq_feature_indices[eq_i] = condition.feature_index
           eq_thresholds[eq_i] = condition.threshold
           eq_i += 1
        else:
           neq_feature_indices[neq_i] = condition.feature_index
           neq_thresholds[neq_i] = condition.threshold
           neq_i += 1

        postincrement(iterator)

    cdef ConjunctiveBody rule_body = ConjunctiveBody.__new__(ConjunctiveBody, leq_feature_indices, leq_thresholds,
                                                             gr_feature_indices, gr_thresholds, eq_feature_indices,
                                                             eq_thresholds, neq_feature_indices, neq_thresholds)
    cdef Head rule_head

    if label_indices is None:
        rule_head = FullHead.__new__(FullHead, predicted_scores)
    else:
        rule_head = PartialHead.__new__(PartialHead, label_indices, predicted_scores)

    return Rule.__new__(Rule, rule_body, rule_head)
