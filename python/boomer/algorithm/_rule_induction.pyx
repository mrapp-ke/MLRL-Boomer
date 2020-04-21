# distutils: language=c++

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement algorithms for the induction of individual classification rules.
"""
from boomer.algorithm._arrays cimport uint32, float64, array_intp, array_float32, matrix_intp
from boomer.algorithm._model cimport Head, FullHead, PartialHead, EmptyBody, ConjunctiveBody
from boomer.algorithm._head_refinement cimport HeadCandidate
from boomer.algorithm._losses cimport Prediction
from boomer.algorithm._utils cimport Comparator, Condition, test_condition, get_index, get_weight

from libcpp.list cimport list as list
from cython.operator cimport dereference, postincrement


cdef class RuleInduction:
    """
    A base class for all classes that implement an algorithm for the induction of individual classification rules.
    """

    cpdef Rule induce_default_rule(self, uint8[::1, :] y, Loss loss):
        """
        Induces the default rule that minimizes a certain loss function with respect to the given ground truth labels.

        :param y:       An array of dtype float, shape `(num_examples, num_labels)`, representing the ground truth
                        labels of the training examples
        :param loss:    The loss function to be minimized
        :return:        The default rule that has been induced
        """
        pass

    cpdef Rule induce_rule(self, intp[::1] nominal_attribute_indices, float32[::1, :] x, intp[::1, :] x_sorted_indices,
                           uint8[::1, :] y, HeadRefinement head_refinement, Loss loss,
                           LabelSubSampling label_sub_sampling, InstanceSubSampling instance_sub_sampling,
                           FeatureSubSampling feature_sub_sampling, Pruning pruning, Shrinkage shrinkage,
                           random_state: int):
        """
        Induces a single- or multi-label classification rule that minimizes a certain loss function for the training
        examples it covers.

        :param nominal_attribute_indices:   An array of dtype int, shape `(num_nominal_attributes)`, representing the
                                            indices of all nominal features (in ascending order) or None, if no nominal
                                            features are available
        :param x:                           An array of dtype float, shape `(num_examples, num_features)`, representing the
                                            features of the training examples
        :param x_sorted_indices:            An array of dtype int, shape `(num_examples, num_features)`, representing the
                                            indices of the training examples when sorting column-wise
        :param y:                           An array of dtype int, shape `(num_examples, num_labels)`, representing the
                                            labels of the training examples
        :param head_refinement:             The strategy that is used to find the heads of rules
        :param loss:                        The loss function to be minimized
        :param label_sub_sampling:          The strategy that should be used to sub-sample the labels or None, if no label
                                            sub-sampling should be used
        :param instance_sub_sampling:       The strategy that should be used to sub-sample the training examples or None, if
                                            no instance sub-sampling should be used
        :param feature_sub_sampling:        The strategy that should be used to sub-sample the available features or None,
                                            if no feature sub-sampling should be used
        :param pruning:                     The strategy that should be used to prune rules or None, if no pruning should be
                                            used
        :param shrinkage:                   The strategy that should be used to shrink the weights of rules or None, if no
                                            shrinkage should be used
        :param random_state:                The seed to be used by RNGs
        :return:                            The rule that has been induced
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

    cpdef Rule induce_default_rule(self, uint8[::1, :] y, Loss loss):
        cdef float64[::1] scores = loss.calculate_default_scores(y)
        cdef FullHead head = FullHead.__new__(FullHead, scores)
        cdef EmptyBody body = EmptyBody.__new__(EmptyBody)
        cdef Rule rule = Rule(body, head)
        return rule

    cpdef Rule induce_rule(self, intp[::1] nominal_attribute_indices, float32[::1, :] x, intp[::1, :] x_sorted_indices,
                           uint8[::1, :] y, HeadRefinement head_refinement, Loss loss, LabelSubSampling label_sub_sampling,
                           InstanceSubSampling instance_sub_sampling, FeatureSubSampling feature_sub_sampling,
                           Pruning pruning, Shrinkage shrinkage, random_state: int):
        # The head of the induced rule
        cdef HeadCandidate head = None
        # A list that contains the rule's conditions (in the order they have been learned)
        cdef list[Condition] conditions
        # An array that specifies the number of conditions that use the different types of operators
        cdef intp[::1] num_conditions_per_comparator = array_intp(4)
        num_conditions_per_comparator[:] = 0
        # The indices of the examples that are covered by the induced rule
        cdef intp[::1] covered_example_indices

        # Variables used to update the seed used by RNGs, depending on the refinement iteration (starting at 1)
        cdef int current_random_state = random_state
        cdef int num_refinements = 1

        # Variables for representing the best refinement
        cdef found_refinement = 1
        cdef Comparator best_condition_comparator
        cdef intp best_condition_start, best_condition_end, best_condition_index
        cdef float32 best_condition_threshold

        # Variables for specifying the examples and labels that should be used for finding the best refinement
        cdef intp[::1, :] sorted_indices = x_sorted_indices
        cdef intp num_examples = x.shape[0]

        # Variables for specifying the features used for finding the best refinement
        cdef intp num_nominal_features = nominal_attribute_indices.shape[0] if nominal_attribute_indices is not None else 0
        cdef intp next_nominal_f = -1
        cdef intp[::1] feature_indices
        cdef intp num_features, next_nominal_c
        cdef bint nominal

        # Temporary variables
        cdef HeadCandidate current_head
        cdef Prediction prediction
        cdef float32 previous_threshold, current_threshold
        cdef uint32 weight
        cdef intp c, f, r, i, first_r, previous_r

        # Sub-sample examples, if necessary...
        cdef uint32[::1] weights

        if instance_sub_sampling is None:
            weights = None

             # Notify the loss that all examples should be considered...
            loss.begin_instance_sub_sampling()

            for i in range(num_examples):
                loss.update_sub_sample(i)
        else:
            weights = instance_sub_sampling.sub_sample(x, loss, random_state)

        # Sub-sample labels, if necessary...
        cdef intp[::1] label_indices

        if label_sub_sampling is None:
            label_indices = None
        else:
            label_indices = label_sub_sampling.sub_sample(y, random_state)

        # Search for the best refinement until no improvement in terms of the rule's quality score is possible anymore...
        while found_refinement:
            num_examples = sorted_indices.shape[0]
            found_refinement = 0

            # Sub-sample features, if necessary...
            if feature_sub_sampling is None:
                feature_indices = None
                num_features = x.shape[1]
            else:
                feature_indices = feature_sub_sampling.sub_sample(x, current_random_state)
                num_features = feature_indices.shape[0]

            # Obtain the index of the first nominal feature, if available...
            if num_nominal_features > 0:
                next_nominal_f = nominal_attribute_indices[0]
                next_nominal_c = 1

            # Search for the best condition among all available features to be added to the current rule. For each feature,
            # the examples are traversed in increasing order of their respective feature values and the loss function is
            # updated accordingly. For each potential condition, a quality score is calculated to keep track of the best
            # possible refinement.
            for c in range(num_features):
                first_r = 0
                f = get_index(c, feature_indices)

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

                # Find first example with weight > 0...
                for r in range(num_examples):
                    i = sorted_indices[r, f]
                    weight = get_weight(i, weights)

                    if weight > 0:
                        # Tell the loss function that the example will be covered by upcoming refinements...
                        loss.update_search(i, weight)
                        previous_threshold = x[i, f]
                        previous_r = r
                        break

                # Traverse remaining instances...
                for r in range(r + 1, num_examples):
                    i = sorted_indices[r, f]
                    weight = get_weight(i, weights)

                    # Do only consider examples that are included in the current sub-sample...
                    if weight > 0:
                        current_threshold = x[i, f]

                        # Split points between examples with the same feature value must not be considered...
                        if previous_threshold != current_threshold:
                            # Find and evaluate the best head for the current refinement, if a condition that uses the <=
                            # operator (or the == operator in case of a nominal feature) is used...
                            current_head = head_refinement.find_head(head, label_indices, loss, False)

                            # If refinement using the <= operator (or the == operator in case of a nominal feature) is
                            # better than the current rule...
                            if current_head is not None:
                                found_refinement = 1
                                head = current_head
                                best_condition_start = first_r
                                best_condition_end = r
                                best_condition_index = f

                                if nominal:
                                    best_condition_comparator = Comparator.EQ
                                    best_condition_threshold = previous_threshold
                                else:
                                    best_condition_comparator = Comparator.LEQ
                                    best_condition_threshold = (previous_threshold + current_threshold) / 2.0

                                # If instance sub-sampling is used, examples that are not contained in the current
                                # sub-sample were not considered for finding the condition. Later on, we need to identify
                                # the examples that are covered by the refined rule, including those that are not contained
                                # in the sub-sample, via the function `_filter_sorted_indices`. Said function calculates the
                                # number of covered examples based on the variable `best_condition_end`, which represents
                                # the position that separates the covered from the uncovered examples. However, when taking
                                # into account the examples that are not contained in the sub-sample, this position may
                                # differ from the value of `best_condition_end` at this point and therefore must be
                                # adjusted...
                                if instance_sub_sampling is not None and r - previous_r > 1:
                                    best_condition_end = __adjust_split(x, sorted_indices, r, previous_r, f,
                                                                        best_condition_threshold)

                            # Find and evaluate the best head for the current refinement, if a condition that uses the >
                            # operator (or the != operator in case of a nominal feature) is used...
                            current_head = head_refinement.find_head(head, label_indices, loss, True)

                            # If refinement using the > operator (or the != operator in case of a nominal feature) is better
                            # than the current rule...
                            if current_head is not None:
                                found_refinement = 1
                                head = current_head
                                best_condition_start = first_r
                                best_condition_end = r
                                best_condition_index = f

                                if nominal:
                                    best_condition_comparator = Comparator.NEQ
                                    best_condition_threshold = previous_threshold
                                else:
                                    best_condition_comparator = Comparator.GR
                                    best_condition_threshold = (previous_threshold + current_threshold) / 2.0

                                # Again, if instance sub-sampling is used, we need to adjust the position that separates the
                                # covered from the uncovered examples, including those that are not contained in the sample
                                # (see description above for details)...
                                if instance_sub_sampling is not None and r - previous_r > 1:
                                    best_condition_end = __adjust_split(x, sorted_indices, r, previous_r, f,
                                                                        best_condition_threshold)

                            # Reset the loss function in case of a nominal feature, as the previous examples will not be
                            # covered by the next condition...
                            if nominal:
                                loss.begin_search(label_indices)
                                first_r = r

                        previous_threshold = current_threshold
                        previous_r = r

                        # Tell the loss function that the example will be covered by upcoming refinements...
                        loss.update_search(i, weight)

            if found_refinement:
                # If a refinement has been found, add the new condition...
                conditions.push_back(__make_condition(best_condition_index, best_condition_comparator,
                                                      best_condition_threshold))
                num_conditions_per_comparator[<intp>best_condition_comparator] += 1

                # Update the examples and labels for which the rule predicts...
                label_indices = head.label_indices
                sorted_indices = __filter_sorted_indices(x, sorted_indices, best_condition_start, best_condition_end,
                                                         best_condition_index, best_condition_comparator,
                                                         best_condition_threshold, loss)

                if num_examples > 1:
                    # Alter seed to be used by RNGs for the next refinement...
                    num_refinements += 1
                    current_random_state = random_state * num_refinements
                else:
                    # Abort refinement process if rule covers a single example...
                    break

        if head is None:
            raise RuntimeError('Failed to find an useful condition for the new rule! Please remove any constants features from the training data')

        # Obtain the indices of all examples that are covered by the new rule, regardless of whether they are included in
        # the sub-sample or not...
        covered_example_indices = sorted_indices[:, 0]

        if weights is not None:
            # Prune rule, if necessary (a rule can only be pruned if it contains more than one condition)...
            if pruning is not None and conditions.size() > 1:
                pruning.begin_pruning(weights, loss, head_refinement, covered_example_indices, label_indices)
                covered_example_indices = pruning.prune(x, x_sorted_indices, conditions)

            # If instance sub-sampling is used, we need to re-calculate the scores in the head based on the entire training
            # data...
            loss.begin_search(label_indices)

            for i in covered_example_indices:
                loss.update_search(i, 1)

            prediction = head_refinement.evaluate_predictions(loss, 0)
            __copy_array(prediction.predicted_scores, head.predicted_scores)

        # Apply shrinkage, if necessary...
        if shrinkage is not None:
            shrinkage.apply_shrinkage(head.predicted_scores)

        # Tell the loss function that a new rule has been induced...
        loss.apply_predictions(covered_example_indices, label_indices, head.predicted_scores)

        # Build and return the induced rule...
        return __build_rule(head, conditions, num_conditions_per_comparator)


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


cdef inline __copy_array(float64[::1] from_array, float64[::1] to_array):
    """
    Copies the elements from one array to another.

    :param from_array:  An array of dtype float, shape `(num_elements)`, representing the array from which the elements
                        should be copied
    :param to_array:    An array of dtype float, shape `(num_elements)`, representing the array to which the elements
                        should be copied
    """
    cdef intp num_elements = from_array.shape[0]
    cdef intp i

    for i in range(num_elements):
        to_array[i] = from_array[i]


cdef inline intp __adjust_split(float32[::1, :] x, intp[::1, :] sorted_indices, intp position_start, intp position_end,
                                intp feature_index, float32 threshold):
   """
   Adjusts the position that separates the covered from the uncovered examples with respect to those examples that are
   not contained in the current sub-sample. This requires to look back a certain number of examples (until the next
   example that is contained in the current sub-sample is encountered) to see if they satisfy the new condition or not.

   :param x:               An array of dtype float, shape `(num_examples, num_features)`, representing the features of
                           the training examples
   :param sorted_indices:  An array of dtype int, shape `(num_examples, num_features)`, representing the indices of the
                           examples that are covered by the previous rule when sorting column-wise
   :param position_start:  The position that separates the covered from the uncovered examples (when only taking into
                           account the examples that are contained in the sample). This is the position to start at
   :param position_end:    The position to stop at (exclusive, must be smaller than `position_start`)
   :param feature_index:   The index of the feature, the condition corresponds to
   :param threshold:       The threshold of the condition
   :return:                The adjusted position that separates the covered from the uncovered examples with respect to
                           the examples that are not contained in the sample
   """
   cdef intp adjusted_position = position_start
   cdef float32 feature_value
   cdef intp r, i

   # Traverse the preceding examples until we encounter an example that is contained in the current sub-sample...
   for r in range(position_start - 1, position_end, -1):
        i = sorted_indices[r, feature_index]
        feature_value = x[i, feature_index]

        if feature_value > threshold:
            # The feature value at `position_start` is guaranteed to be greater than the given `threshold`. If this does
            # also apply to the feature value of a preceding example, it is not separated from the example at
            # `position_start`. Hence, we are not done yet and continue by decrementing the adjusted position by one...
            adjusted_position = r
        else:
            # If we have found the first example that is separated from the example at the position we started at, we
            # are done...
            break

   return adjusted_position


cdef inline Rule __build_rule(HeadCandidate head, list[Condition] conditions,
                                  intp[::1] num_conditions_per_comparator):
    """
    Builds and returns a rule.

    :param head:                            A 'HeadCandidate' representing the head of the rule
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

    if head.label_indices is None:
        rule_head = FullHead.__new__(FullHead, head.predicted_scores)
    else:
        rule_head = PartialHead.__new__(PartialHead, head.label_indices, head.predicted_scores)

    cdef Rule rule = Rule.__new__(Rule, rule_body, rule_head)
    return rule


cdef inline intp[::1, :] __filter_sorted_indices(float32[::1, :] x, intp[::1, :] sorted_indices, intp condition_start,
                                                 intp condition_end, intp condition_index,
                                                 Comparator condition_comparator, float32 condition_threshold,
                                                 Loss loss):
    """
    Filters the matrix of example indices after a new condition has been added to a previous rule, such that the
    filtered matrix does only contain the indices of examples that are covered by the new rule.

    :param x:                       An array of dtype float, shape `(num_examples, num_features)`, representing the
                                    features of the training examples
    :param sorted_indices:          An array of dtype int, shape `(num_examples, num_features)`, representing the
                                    indices of the training examples that are covered by the previous rule when sorting
                                    column-wise
    :param condition_start:         The row in `sorted_indices` that corresponds to the first example (inclusive) that
                                    has been passed to the loss function when searching for the new condition
    :param condition_end:           The row in `sorted_indices` that corresponds to the last example (exclusive) that
                                    has been passed to the loss function when searching for the new condition
    :param condition_index:         The index of the feature, the new condition corresponds to
    :param condition_comparator:    The type of the operator that is used by the new condition
    :param condition_threshold:     The threshold of the new condition
    :param loss:                    The loss function to be notified about the examples that must be considered when
                                    searching for the next refinement, i.e., the examples that are covered by the
                                    current rule
    :return:                        An array of dtype int, shape `(num_covered_examples, num_features)`, representing
                                    the indices of the examples that are covered by the new rule when sorting
                                    column-wise
    """
    cdef intp num_features = x.shape[1]
    cdef intp num_examples = sorted_indices.shape[0]
    cdef intp num_covered = condition_end - condition_start
    cdef intp first, last

    if condition_comparator == Comparator.GR or condition_comparator == Comparator.NEQ:
        num_covered = num_examples - num_covered
        first = condition_end
        last = num_examples
    else:
        first = condition_start
        last = condition_end

    cdef intp[::1, :] filtered_sorted_indices = matrix_intp(num_covered, num_features)
    cdef float32 feature_value
    cdef intp c, r, i, index

    # Tell the loss function that a new sub-sample of examples will be selected...
    loss.begin_instance_sub_sampling()

    for c in range(num_features):
        i = 0

        if c == condition_index:
            # For the feature that corresponds to the new condition we know the indices of the covered examples...
            if condition_comparator == Comparator.NEQ:
                for r in range(condition_start):
                    index = sorted_indices[r, c]
                    filtered_sorted_indices[i, c] = index
                    i += 1

                    # Tell the loss function that the example at the current index is covered by the current rule...
                    loss.update_sub_sample(index)

            for r in range(first, last):
                index = sorted_indices[r, c]
                filtered_sorted_indices[i, c] = index
                i += 1

                # Tell the loss function that the example at the current index is covered by the current rule...
                loss.update_sub_sample(index)
        else:
            # For the other features we need to filter out the indices that correspond to examples that do not satisfy
            # the new condition...
            for r in range(num_examples):
                index = sorted_indices[r, c]
                feature_value = x[index, condition_index]

                if test_condition(condition_threshold, condition_comparator, feature_value):
                    filtered_sorted_indices[i, c] = index
                    i += 1

                    if i >= num_covered:
                        break

    return filtered_sorted_indices