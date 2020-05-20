# distutils: language=c++

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement algorithms for inducing individual classification rules.
"""
from boomer.algorithm._arrays cimport uint32, float64, array_intp, array_float32, matrix_intp, get_index
from boomer.algorithm.rules cimport Head, FullHead, PartialHead, EmptyBody, ConjunctiveBody
from boomer.algorithm.head_refinement cimport HeadCandidate
from boomer.algorithm.losses cimport Prediction

from libc.stdlib cimport qsort

from libcpp.list cimport list
from libcpp.pair cimport pair

from cython.operator cimport dereference, postincrement

from cpython.mem cimport PyMem_Malloc as malloc, PyMem_Realloc as realloc, PyMem_Free as free


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

    cdef Rule induce_rule(self, intp[::1] nominal_attribute_indices, float32[::1] x_data, intp[::1] x_row_indices,
                          intp[::1] x_col_indices, uint8[::1, :] y, HeadRefinement head_refinement, Loss loss,
                          LabelSubSampling label_sub_sampling, InstanceSubSampling instance_sub_sampling,
                          FeatureSubSampling feature_sub_sampling, Pruning pruning, Shrinkage shrinkage,
                          intp min_coverage, intp max_condition, RNG rng):
        """
        Induces a single- or multi-label classification rule that minimizes a certain loss function for the training
        examples it covers.

        :param nominal_attribute_indices:   An array of dtype int, shape `(num_nominal_attributes)`, representing the
                                            indices of all nominal features (in ascending order) or None, if no nominal
                                            features are available
        :param x_data:                      An array of dtype float, shape `(num_non_zero_feature_values)`, representing
                                            the non-zero feature values of the training examples
        :param x_row_indices:               An array of dtype int, shape `(num_non_zero_feature_values)`, representing
                                            the row-indices of the examples, the values in `x_data` correspond to
        :param x_col_indices:               An array of dtype int, shape `(num_features + 1)`, representing the indices
                                            of the first element in `x_data` and `x_row_indices` that corresponds to a
                                            certain feature. The index at the last position is equal to
                                            `num_non_zero_feature_values`
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
        self.sorted_indices_map_global = new map[intp, IndexedArray*]()

    def __dealloc__(self):
        cdef map[intp, IndexedArray*]* sorted_indices_map_global = self.sorted_indices_map_global
        cdef map[intp, IndexedArray*].iterator iterator = dereference(sorted_indices_map_global).begin()
        cdef IndexedArray* indexed_array

        while iterator != dereference(sorted_indices_map_global).end():
            indexed_array = dereference(iterator).second
            free(dereference(indexed_array).data)
            free(indexed_array)
            postincrement(iterator)

        del self.sorted_indices_map_global

    cdef Rule induce_default_rule(self, uint8[::1, :] y, Loss loss):
        cdef float64[::1] scores = loss.calculate_default_scores(y)
        cdef FullHead head = FullHead.__new__(FullHead, scores)
        cdef EmptyBody body = EmptyBody.__new__(EmptyBody)
        cdef Rule rule = Rule.__new__(Rule, body, head)
        return rule

    cdef Rule induce_rule(self, intp[::1] nominal_attribute_indices, float32[::1] x_data, intp[::1] x_row_indices,
                          intp[::1] x_col_indices, uint8[::1, :] y, HeadRefinement head_refinement, Loss loss,
                          LabelSubSampling label_sub_sampling, InstanceSubSampling instance_sub_sampling,
                          FeatureSubSampling feature_sub_sampling, Pruning pruning, Shrinkage shrinkage,
                          intp min_coverage, intp max_conditions, RNG rng):
        # The total number of examples in the training data set
        cdef intp num_examples = x_data.shape[0]
        # The total number of features in the training data set
        cdef intp num_features = x_col_indices.shape[0] - 1
        # The total number of labels in the training data set
        cdef intp num_labels = y.shape[1]
        # The head of the induced rule
        cdef HeadCandidate head = None
        # A (stack-allocated) list that contains the conditions in the rule's body (in the order they have been learned)
        cdef list[Condition] conditions
        # The total number of conditions
        cdef intp num_conditions = 0
        # An array representing the number of conditions per type of operator
        cdef intp[::1] num_conditions_per_comparator = array_intp(4)
        num_conditions_per_comparator[:] = 0
        # An array representing the indices of the examples that are covered by the rule
        cdef intp[::1] covered_example_indices = None

        # Variables for representing the best refinement
        cdef bint found_refinement = True
        cdef Comparator best_condition_comparator
        cdef intp best_condition_start, best_condition_end, best_condition_previous, best_condition_index
        cdef float32 best_condition_threshold
        cdef intp best_condition_covered_weights
        cdef IndexedValue* best_condition_indexed_values
        cdef IndexedArrayWrapper* best_condition_indexed_array_wrapper

        # Variables for specifying the examples that should be used for finding the best refinement
        cdef map[intp, IndexedArray*]* sorted_indices_map_global = self.sorted_indices_map_global
        cdef IndexedArray* indexed_array
        cdef map[intp, IndexedArrayWrapper*] sorted_indices_map_local  # Stack-allocated map
        cdef map[intp, IndexedArrayWrapper*].iterator indexed_array_wrapper_iterator
        cdef IndexedArrayWrapper* indexed_array_wrapper
        cdef IndexedValue* indexed_values
        cdef intp num_covered = num_examples
        cdef intp num_non_zero_feature_values

        # Variables for specifying the features used for finding the best refinement
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
        cdef intp total_sum_of_weights, sum_of_weights

        if instance_sub_sampling is None:
            weights = None
            total_sum_of_weights = num_examples
        else:
            instance_sub_sampling_result = instance_sub_sampling.sub_sample(num_examples, rng)
            weights = instance_sub_sampling_result.first
            total_sum_of_weights = instance_sub_sampling_result.second

        loss.set_sub_sample(covered_example_indices, weights)

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
                    indexed_array_wrapper = sorted_indices_map_local[f]

                    if indexed_array_wrapper == NULL:
                        indexed_array_wrapper = <IndexedArrayWrapper*>malloc(sizeof(IndexedArrayWrapper))
                        dereference(indexed_array_wrapper).array = NULL
                        dereference(indexed_array_wrapper).num_conditions = 0
                        sorted_indices_map_local[f] = indexed_array_wrapper

                    indexed_array = dereference(indexed_array_wrapper).array

                    if indexed_array == NULL:
                        indexed_array = dereference(sorted_indices_map_global)[f]

                        if indexed_array == NULL:
                            indexed_array = __argsort_by_feature_values(x_data, x_row_indices, x_col_indices, f)
                            dereference(sorted_indices_map_global)[f] = indexed_array

                    # Filter indices, if only a subset of the contained examples is covered...
                    if num_conditions > dereference(indexed_array_wrapper).num_conditions:
                        # TODO Check arguments
                        __filter_any_indices(x, sorted_indices, num_examples, index_array, conditions, num_conditions,
                                             num_covered)
                        indexed_array = dereference(indexed_array_wrapper).array

                    num_non_zero_feature_values = dereference(indexed_array).num_elements
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
                    first_r = num_non_zero_feature_values - 1

                    # Traverse examples in descending order until the first example with weight > 0 is encountered...
                    for r in range(num_non_zero_feature_values - 1, -1, -1):
                        i = indexed_values[r].index
                        weight = 1 if weights is None else weights[i]

                        if weight > 0:
                            # Tell the loss function that the example will be covered by upcoming refinements...
                            loss.update_search(i, weight)
                            sum_of_weights += weight
                            previous_threshold = indexed_values[r].value
                            previous_r = r
                            break

                    # Traverse the remaining examples in descending order...
                    # TODO explicitly handle zero feature values
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
                                current_head = head_refinement.find_head(head, label_indices, loss, False)

                                # If refinement using the > operator (or the == operator in case of a nominal feature)
                                # is better than the current rule...
                                if current_head is not None:
                                    found_refinement = True
                                    head = current_head
                                    best_condition_start = first_r
                                    best_condition_end = r
                                    best_condition_previous = previous_r
                                    best_condition_index = f
                                    best_condition_covered_weights = sum_of_weights
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
                                current_head = head_refinement.find_head(head, label_indices, loss, True)

                                # If refinement using the <= operator (or the != operator in case of a nominal feature)
                                # is better than the current rule...
                                if current_head is not None:
                                    found_refinement = True
                                    head = current_head
                                    best_condition_start = first_r
                                    best_condition_end = r
                                    best_condition_previous = previous_r
                                    best_condition_index = f
                                    best_condition_covered_weights = (total_sum_of_weights - sum_of_weights)
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
                                    loss.begin_search(label_indices)
                                    sum_of_weights = 0
                                    first_r = r

                            previous_threshold = current_threshold
                            previous_r = r

                            # Tell the loss function that the example will be covered by upcoming refinements...
                            loss.update_search(i, weight)
                            sum_of_weights += weight

                if found_refinement:
                    # If a refinement has been found, add the new condition and update the labels for which the rule
                    # predicts...
                    conditions.push_back(__make_condition(best_condition_index, best_condition_comparator,
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
                                                            best_condition_previous, best_condition_index,
                                                            best_condition_threshold)

                    # Identify the examples for which the rule predicts...
                    # TODO Check arguments
                    __filter_current_indices(best_condition_sorted_indices, num_examples, best_condition_index_array,
                                             best_condition_start, best_condition_end, best_condition_index,
                                             best_condition_comparator, num_conditions)
                    num_covered = dereference(best_condition_index_array).num_elements
                    covered_example_indices = <intp[:num_covered]>dereference(best_condition_index_array).data

                    if best_condition_covered_weights > min_coverage:
                        # Inform the loss function about the weights of the examples that are covered by the current
                        # rule...
                        loss.set_sub_sample(covered_example_indices, weights)
                        total_sum_of_weights = best_condition_covered_weights
                    else:
                        # Abort refinement process if rule covers a single example...
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
                        pruning.begin_pruning(weights, loss, head_refinement, covered_example_indices, label_indices)
                        covered_example_indices = pruning.prune(x, sorted_indices_map_global, conditions)
                        num_covered = covered_example_indices.shape[0]

                    # If instance sub-sampling is used, we need to re-calculate the scores in the head based on the
                    # entire training data...
                    # TODO revise re-calculation of predicted scores
                    loss.begin_search(label_indices)

                    for r in range(num_covered):
                        i = covered_example_indices[r]
                        loss.update_search(i, 1)

                    prediction = head_refinement.evaluate_predictions(loss, False)
                    predicted_scores[:] = prediction.predicted_scores

                # Apply shrinkage, if necessary...
                if shrinkage is not None:
                    shrinkage.apply_shrinkage(predicted_scores)

                # Tell the loss function that a new rule has been induced...
                loss.apply_predictions(covered_example_indices, label_indices, predicted_scores)

                # Build and return the induced rule...
                return __build_rule(label_indices, predicted_scores, conditions, num_conditions_per_comparator)
        finally:
            # Free memory occupied by the arrays stored in `sorted_indices_map_local`...
            indexed_array_wrapper_iterator = sorted_indices_map_local.begin()

            while indexed_array_wrapper_iterator != sorted_indices_map_local.end():
                indexed_array_wrapper = dereference(indexed_array_wrapper_iterator).second
                indexed_array = dereference(indexed_array_wrapper).array
                indexed_values = dereference(indexed_array).data
                free(indexed_values)
                free(indexed_array)
                free(indexed_array_wrapper)
                postincrement(indexed_array_wrapper_iterator)


cdef inline IndexedArray* __argsort_by_feature_values(float32[::1] x_data, intp[::1] x_row_indices,
                                                      intp[::1] x_col_indices, intp feature_index):
    """
    Creates and returns a pointer to a struct of type `IndexedArray` that stores the indices of the training examples,
    as well as their feature values for a specific feature, sorted in ascending order by the feature values.

    :param x_data:          An array of dtype float, shape `(num_non_zero_feature_values)`, representing the non-zero
                            feature values of the training examples
    :param x_row_indices:   An array of dtype int, shape `(num_non_zero_feature_values)`, representing the row-indices
                            of the examples, the values in `x_data` correspond to
    :param x_col_indices:   An array of dtype int, shape `(num_features + 1)`, representing the indices of the first
                            element in `x_data` and `x_row_indices` that corresponds to a certain feature. The index at
                            the last position is equal to `num_non_zero_feature_values`
    :param feature_index:   The index of the feature, the feature values that should be used for sorting correspond to
    :return:                A pointer to a struct of type `IndexedArray`
    """
    cdef intp start = x_col_indices[feature_index]
    cdef intp end = x_col_indices[feature_index + 1]
    cdef intp num_values = end - start
    cdef IndexedArray* indexed_array = <IndexedArray*>malloc(sizeof(IndexedArray))
    dereference(indexed_array).num_elements = num_values

    if num_values > 0:
        cdef IndexedValue* sorted_array = <IndexedValue*>malloc(num_values * sizeof(IndexedValue))
        cdef intp i = 0
        cdef intp j

        for j in range(start, end):
            sorted_array[i].index = x_row_indices[j]
            sorted_array[i].value = x_data[j]
            i += 1

        qsort(sorted_array, num_values, sizeof(IndexedValue), &__compare_indexed_value)
        dereference(indexed_array).data = sorted_array
    else:
        dereference(indexed_array).data = NULL

    return indexed_array


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
                                intp feature_index, float32 threshold):
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
    :param feature_index:   The index of the feature, the condition corresponds to
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


cdef inline void __filter_current_indices(intp* sorted_indices, intp num_indices, IndexArray* index_array,
                                          intp condition_start, intp condition_end, intp condition_index,
                                          Comparator condition_comparator, intp num_conditions):
    """
    Filters an array that contains the indices of the examples that are covered by the previous rule after a new
    condition has been added, such that the filtered array does only contain the indices of the examples that are
    covered by the new rule. The filtered array is stored in a given struct of type `IndexArray`.

    :param sorted_indices:          A pointer to a C-array of type int, shape `(num_indices)`, representing the indices
                                    of the training examples that are covered by the previous rule in ascending order of
                                    values for the feature, the new condition corresponds to
    :param num_indices:             The number of elements in the array `sorted_indices`
    :param index_array:             A pointer to a struct of type `IndexArray` that should be used to store the filtered
                                    array
    :param condition_start:         The element in `sorted_indices` that corresponds to the first example (inclusive)
                                    that has been passed to the loss function when searching for the new condition (must
                                    be greater than `condition_end`)
    :param condition_end:           The element in `sorted_indices_map[condition_index]` that corresponds to the last
                                    example (exclusive) that has been passed to the loss function when searching for the
                                    new condition (must be smaller than `condition_start`)
    :param condition_index:         The index of the feature, the new condition corresponds to
    :param condition_comparator:    The type of the operator that is used by the new condition
    :param num_conditions:          The total number of conditions in the rule's body (including the new one)
    """
    cdef intp num_covered = condition_start - condition_end
    cdef intp r, first, last, index

    if condition_comparator == Comparator.LEQ or condition_comparator == Comparator.NEQ:
        num_covered = num_indices - num_covered
        first = condition_end
        last = -1
    else:
        first = condition_start
        last = condition_end

    cdef intp* filtered_indices_array = <intp*>malloc(num_covered * sizeof(intp))
    cdef intp i = num_covered - 1

    if condition_comparator == Comparator.NEQ:
        for r in range(num_indices - 1, condition_start, -1):
            index = sorted_indices[r]
            filtered_indices_array[i] = index
            i -= 1

    for r in range(first, last, -1):
        index = sorted_indices[r]
        filtered_indices_array[i] = index
        i -= 1

    free(dereference(index_array).data)
    dereference(index_array).data = filtered_indices_array
    dereference(index_array).num_elements = num_covered
    dereference(index_array).num_conditions = num_conditions


cdef inline void __filter_any_indices(float32[::1, :] x, intp* sorted_indices, intp num_indices,
                                      IndexArray* index_array, list[Condition] conditions, intp num_conditions,
                                      intp num_covered):
    """
    Filters an array that contains the indices of examples with respect to one or several conditions, such that the
    filtered array does only contain the indices of the examples that satisfy the conditions. The filtered array is
    stored in a given struct of type `IndexArray`.

    :param x:                       An array of dtype float, shape `(num_examples, num_features)`, representing the
                                    features of the training examples
    :param sorted_indices:          A pointer to a C-array of type int, shape `(num_indices)`, representing the indices
                                    of the training examples
    :param num_indices:             The number of elements in the array `sorted_indices`
    :param index_array:             A pointer to a struct of type `IndexArray` that should be used to store the filtered
                                    array
    :param conditions:              A list that contains the conditions that should be taken into account for filtering
                                    the indices
    :param num_conditions:          The number of conditions in the list `conditions`
    :param num_covered:             The number of training examples that satisfy all conditions in the list `conditions`
    """
    cdef intp* filtered_indices_array = dereference(index_array).data
    cdef bint must_allocate = filtered_indices_array == NULL

    if must_allocate:
        filtered_indices_array = <intp*>malloc(num_covered * sizeof(intp))

    cdef intp num_untested_conditions = num_conditions - dereference(index_array).num_conditions
    cdef intp i = 0
    cdef list[Condition].reverse_iterator iterator
    cdef Condition condition
    cdef Comparator condition_comparator
    cdef float32 condition_threshold, feature_value
    cdef intp condition_index, c, r, index
    cdef bint covered

    for r in range(num_indices):
        index = sorted_indices[r]
        covered = True

        # Traverse conditions in reverse order...
        iterator = conditions.rbegin()
        c = 0

        while c < num_untested_conditions:
            condition = dereference(iterator)
            condition_threshold = condition.threshold
            condition_comparator = condition.comparator
            condition_index = condition.feature_index
            feature_value = x[index, condition_index]

            if not test_condition(condition_threshold, condition_comparator, feature_value):
                covered = False
                break

            c += 1
            postincrement(iterator)

        if covered:
            filtered_indices_array[i] = index
            i += 1

            if i >= num_covered:
                break

    if not must_allocate:
        filtered_indices_array = <intp*>realloc(filtered_indices_array, num_covered * sizeof(intp))

    dereference(index_array).data = filtered_indices_array
    dereference(index_array).num_elements = num_covered
    dereference(index_array).num_conditions = num_conditions


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
