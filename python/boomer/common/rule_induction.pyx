"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement algorithms for inducing individual classification rules.
"""
from boomer.common._arrays cimport float64, array_uint32
from boomer.common._tuples cimport IndexedFloat32, IndexedFloat32ArrayWrapper
from boomer.common._predictions cimport Prediction, PredictionCandidate
from boomer.common.rules cimport Condition, Comparator
from boomer.common.rule_refinement cimport Refinement, IRuleRefinement, ExactRuleRefinementImpl
from boomer.common.statistics cimport AbstractStatistics, IStatisticsSubset
from boomer.common.sub_sampling cimport IWeightVector, IIndexVector
from boomer.common.thresholds cimport IThresholdsSubset, ExactThresholdsImpl, ThresholdsSubsetImpl

from libc.math cimport fabs
from libc.stdlib cimport abs, malloc, realloc, free

from libcpp.unordered_map cimport unordered_map
from libcpp.list cimport list as double_linked_list
from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr
from libcpp.cast cimport dynamic_cast

from cython.operator cimport dereference, postincrement
from cython.parallel cimport prange


ctypedef ThresholdsSubsetImpl* ThresholdsSubsetImplPtr

cdef class RuleInduction:
    """
    A base class for all classes that implement an algorithm for the induction of individual classification rules.
    """

    cdef void induce_default_rule(self, StatisticsProvider statistics_provider, IHeadRefinement* head_refinement,
                                  ModelBuilder model_builder):
        """
        Induces the default rule.

        :param statistics_provider: A `StatisticsProvider` that provides access to the statistics which should serve as
                                    the basis for inducing the default rule
        :param head_refinement:     A pointer to an object of type `IHeadRefinement` that should be used to find the
                                    head of the default rule or NULL, if no default rule should be induced
        :param model_builder:       The builder, the default rule should be added to
        """
        pass

    cdef bint induce_rule(self, StatisticsProvider statistics_provider, AbstractThresholds* thresholds,
                          INominalFeatureVector* nominal_feature_vector, IFeatureMatrix* feature_matrix,
                          IHeadRefinement* head_refinement, ILabelSubSampling* label_sub_sampling,
                          IInstanceSubSampling* instance_sub_sampling, IFeatureSubSampling* feature_sub_sampling,
                          Pruning pruning, PostProcessor post_processor, uint32 min_coverage, intp max_conditions,
                          intp max_head_refinements, int num_threads, RNG* rng, ModelBuilder model_builder):
        """
        Induces a new classification rule.

        :param statistics_provider:     A `StatisticsProvider` that provides access to the statistics which should serve
                                        as the basis for inducing the new rule
        :param thresholds:              A pointer to an object of type `AbstractThresholds` that provides access to the
                                        thresholds that may be used by the conditions of rules
        :param nominal_feature_vector:  A pointer to an object of type `INominalFeatureVector` that provides access to
                                        the information whether individual features are nominal or not
        :param feature_matrix:          A pointer to an object of type `IFeatureMatrix` that provides column-wise access
                                        to the feature values of the training examples
        :param head_refinement:         A pointer to an object of type `IHeadRefinement` that should be used to find the
                                        head of the rule
        :param label_sub_sampling:      A pointer to an object of type `ILabelSubSampling`, implementing the strategy
                                        that should be used to sub-sample the labels
        :param instance_sub_sampling:   A pointer to an object of type `IInstanceSubSampling`, implementing the strategy
                                        that should be used to sub-sample the training examples
        :param feature_sub_sampling:    A pointer to an object of type `IFeatureSubSampling`, implementing the strategy
                                        that should be used to sub-sample the available features
        :param pruning:                 The strategy that should be used to prune rules or None, if no pruning should be
                                        used
        :param post_processor:          The post-processor that should be used to post-process the rule once it has been
                                        learned or None, if no post-processing should be used
        :param min_coverage:            The minimum number of training examples that must be covered by the rule. Must
                                        be at least 1
        :param max_conditions:          The maximum number of conditions to be included in the rule's body. Must be at
                                        least 1 or -1, if the number of conditions should not be restricted
        :param max_head_refinements:    The maximum number of times the head of a rule may be refined after a new
                                        condition has been added to its body. Must be at least 1 or -1, if the number of
                                        refinements should not be restricted
        :param num_threads:             The number of threads to be used for evaluating the potential refinements of the
                                        rule in parallel. Must be at least 1
        :param rng:                     A pointer to an object of type `RNG`, implementing the random number generator
                                        to be used
        :param model_builder:           The builder, the rule should be added to
        :return:                        1, if a rule has been induced, 0 otherwise
        """
        pass


cdef class TopDownGreedyRuleInduction(RuleInduction):
    """
    Allows to induce single- or multi-label classification rules using a top-down greedy search, where new conditions
    are added iteratively to the (initially empty) body of a rule. At each iteration, the refinement that improves the
    rule the most is chosen. The search stops if no refinement results in an improvement.
    """

    cdef void induce_default_rule(self, StatisticsProvider statistics_provider, IHeadRefinement* head_refinement,
                                  ModelBuilder model_builder):
        cdef unique_ptr[PredictionCandidate] default_prediction_ptr
        cdef unique_ptr[IStatisticsSubset] statistics_subset_ptr
        cdef AbstractStatistics* statistics
        cdef uint32 num_statistics, i

        if head_refinement != NULL:
            statistics = statistics_provider.get()
            num_statistics = statistics.getNumRows()
            statistics.resetSampledStatistics()

            for i in range(num_statistics):
                statistics.addSampledStatistic(i, 1)

            statistics_subset_ptr.reset(statistics.createSubset(0, NULL))
            default_prediction_ptr.reset(head_refinement.findHead(NULL, NULL, NULL, statistics_subset_ptr.get(), True,
                                                                  False))

            statistics_provider.switch_rule_evaluation()

            for i in range(num_statistics):
                statistics.applyPrediction(i, default_prediction_ptr.get())

            model_builder.set_default_rule(default_prediction_ptr.get())
        else:
            statistics_provider.switch_rule_evaluation()

    cdef bint induce_rule(self, StatisticsProvider statistics_provider, AbstractThresholds* thresholds,
                          INominalFeatureVector* nominal_feature_vector, IFeatureMatrix* feature_matrix,
                          IHeadRefinement* head_refinement, ILabelSubSampling* label_sub_sampling,
                          IInstanceSubSampling* instance_sub_sampling, IFeatureSubSampling* feature_sub_sampling,
                          Pruning pruning, PostProcessor post_processor, uint32 min_coverage, intp max_conditions,
                          intp max_head_refinements, int num_threads, RNG* rng, ModelBuilder model_builder):
        # The statistics
        cdef AbstractStatistics* statistics = statistics_provider.get()
        # The total number of statistics
        cdef uint32 num_statistics = thresholds.getNumRows()
        # The total number of features
        cdef uint32 num_features = thresholds.getNumCols()
        # The total number of labels
        cdef uint32 num_labels = thresholds.getNumLabels()
        # A (stack-allocated) list that contains the conditions in the rule's body (in the order they have been learned)
        cdef double_linked_list[Condition] conditions
        # The total number of conditions
        cdef uint32 num_conditions = 0
        # An array representing the number of conditions per type of operator
        cdef uint32[::1] num_conditions_per_comparator = array_uint32(4)
        num_conditions_per_comparator[:] = 0
        # A map that stores a pointer to an object of type `IRuleRefinement` for each feature
        cdef unordered_map[uint32, IRuleRefinement*] rule_refinements  # Stack-allocated map
        # A map that stores the best refinement for each feature
        cdef unordered_map[uint32, Refinement] refinements  # Stack-allocated map
        # The best refinement of the current rule
        cdef Refinement best_refinement  # Stack-allocated struct
        best_refinement.head = NULL
        # Whether a refinement of the current rule has been found
        cdef bint found_refinement = True

        # Temporary variables
        cdef IRuleRefinement* current_rule_refinement
        cdef Refinement current_refinement
        cdef unique_ptr[IIndexVector] sampled_feature_indices_ptr
        cdef uint32 num_sampled_features, weight, f, r
        cdef bint nominal
        cdef intp c

        cdef ExactThresholdsImpl* outer_thresholds
        cdef ThresholdsSubsetImpl* inner_thresholds

        # Sub-sample examples...
        cdef unique_ptr[IWeightVector] weights_ptr
        weights_ptr.reset(instance_sub_sampling.subSample(num_statistics, rng))
        cdef uint32 total_sum_of_weights = weights_ptr.get().getSumOfWeights()

        # Create a new subset of the given thresholds...
        cdef unique_ptr[IThresholdsSubset] thresholds_subset_ptr
        thresholds_subset_ptr.reset(thresholds.createSubset(weights_ptr.get()))

        outer_thresholds = <ExactThresholdsImpl*>thresholds
        inner_thresholds = dynamic_cast[ThresholdsSubsetImplPtr](thresholds_subset_ptr.get())

        # Sub-sample labels...
        cdef unique_ptr[IIndexVector] sampled_label_indices_ptr
        sampled_label_indices_ptr.reset(label_sub_sampling.subSample(num_labels, rng))
        # TODO Reactivate label sampling
        # cdef IIndexVector* label_indices = sampled_label_indices_ptr.get()
        cdef const uint32* label_indices = <const uint32*>NULL
        cdef uint32 num_predictions = 0

        try:
            # Search for the best refinement until no improvement in terms of the rule's quality score is possible
            # anymore or the maximum number of conditions has been reached...
            while found_refinement and (max_conditions == -1 or num_conditions < max_conditions):
                found_refinement = False

                # Sub-sample features...
                sampled_feature_indices_ptr.reset(feature_sub_sampling.subSample(num_features, rng))
                num_sampled_features = sampled_feature_indices_ptr.get().getNumElements()

                # For each feature, create an object of type `IRuleRefinement` and put it into `rule_refinements`...
                for c in range(num_sampled_features):
                    f = sampled_feature_indices_ptr.get().getIndex(<uint32>c)
                    rule_refinements[f] = thresholds_subset_ptr.get().createRuleRefinement(f)

                # Search for the best condition among all available features to be added to the current rule...
                for c in prange(num_sampled_features, nogil=True, schedule='dynamic', num_threads=num_threads):
                    f = sampled_feature_indices_ptr.get().getIndex(<uint32>c)
                    current_rule_refinement = rule_refinements[f]
                    # TODO current_refinement = current_rule_refinement.findRefinement(head_refinement, best_refinement.head, num_predictions, label_indices)
                    nominal = nominal_feature_vector.getValue(f)
                    current_refinement = __find_refinement(f, nominal, num_predictions, label_indices,
                                                           weights_ptr.get(), total_sum_of_weights,
                                                           outer_thresholds.cache_,
                                                           inner_thresholds.cacheFiltered_,
                                                           feature_matrix, inner_thresholds.coveredExamplesMask_,
                                                           inner_thresholds.coveredExamplesTarget_, num_conditions, statistics,
                                                           head_refinement, best_refinement.head)
                    del current_rule_refinement

                    with gil:
                        refinements[f] = current_refinement

                # Pick the best refinement among the refinements that have been found for the different features...
                for c in range(num_sampled_features):
                    f = sampled_feature_indices_ptr.get().getIndex(<uint32>c)
                    current_refinement = refinements[f]

                    if current_refinement.head != NULL and (best_refinement.head == NULL
                                                            or current_refinement.head.overallQualityScore_ < best_refinement.head.overallQualityScore_):
                        del best_refinement.head
                        best_refinement = current_refinement
                        found_refinement = True
                    else:
                        del current_refinement.head

                refinements.clear()

                if found_refinement:
                    # If a refinement has been found, add the new condition...
                    conditions.push_back(__make_condition(best_refinement.featureIndex, best_refinement.comparator,
                                                          best_refinement.threshold))
                    num_conditions += 1
                    num_conditions_per_comparator[<uint32>best_refinement.comparator] += 1

                    if max_head_refinements > 0 and num_conditions >= max_head_refinements:
                        # Keep the labels for which the rule predicts, if the head should not be further refined...
                        num_predictions = best_refinement.head.numPredictions_
                        label_indices = best_refinement.head.labelIndices_

                    # Filter the current subset of thresholds by applying the best refinement that has been found...
                    thresholds_subset_ptr.get().applyRefinement(best_refinement)

                    # Identify the examples for which the rule predicts...
                    # TODO Remove
                    inner_thresholds.coveredExamplesTarget_ = __filter_current_indices(inner_thresholds.cacheFiltered_,
                                                                         best_refinement.featureIndex,
                                                                         best_refinement.indexedArray,
                                                                         best_refinement.start, best_refinement.end,
                                                                         best_refinement.comparator,
                                                                         best_refinement.covered, num_conditions,
                                                                         inner_thresholds.coveredExamplesMask_,
                                                                         inner_thresholds.coveredExamplesTarget_, statistics,
                                                                         weights_ptr.get())
                    total_sum_of_weights = best_refinement.coveredWeights

                    if total_sum_of_weights <= min_coverage:
                        # Abort refinement process if the rule is not allowed to cover less examples...
                        break

            if best_refinement.head == NULL:
                # No rule could be induced, because no useful condition could be found. This might be the case, if all
                # examples have the same values for the considered features.
                return False
            else:
                if weights_ptr.get().hasZeroElements():
                    # TODO Reactivate pruning
                    # Prune rule, if necessary (a rule can only be pruned if it contains more than one condition)...
                    # if pruning is not None and num_conditions > 1:
                    #     uint32_array_scalar_pair = pruning.prune(cache_global, conditions, best_refinement.head,
                    #                                              covered_statistics_mask, covered_statistics_target,
                    #                                              weights_ptr.get(), statistics, head_refinement)
                    #     covered_statistics_mask = uint32_array_scalar_pair.first
                    #     covered_statistics_target = uint32_array_scalar_pair.second

                    # If instance sub-sampling is used, we must re-calculate the scores in the head based on the entire
                    # training data...
                    thresholds_subset_ptr.get().recalculatePrediction(head_refinement, best_refinement)

                # Apply post-processor, if necessary...
                if post_processor is not None:
                    post_processor.post_process(best_refinement.head)

                # Update the statistics by applying the predictions of the new rule...
                thresholds_subset_ptr.get().applyPrediction(best_refinement.head)

                # Add the induced rule to the model...
                model_builder.add_rule(best_refinement.head, conditions, num_conditions_per_comparator)
                return True
        finally:
            del best_refinement.head


cdef Refinement __find_refinement(uint32 feature_index, bint nominal, uint32 num_label_indices,
                                  const uint32* label_indices, IWeightVector* weights, uint32 total_sum_of_weights,
                                  unordered_map[uint32, IndexedFloat32Array*] &cache_global,
                                  unordered_map[uint32, IndexedFloat32ArrayWrapper*] &cache_local,
                                  IFeatureMatrix* feature_matrix, uint32* covered_statistics_mask,
                                  uint32 covered_statistics_target, uint32 num_conditions,
                                  AbstractStatistics* statistics, IHeadRefinement* head_refinement,
                                  PredictionCandidate* head) nogil:
    """
    Finds and returns the best refinement of an existing rule, which results from adding a new condition that
    corresponds to a certain feature.

    :param feature_index:               The index of the feature, the new condition should correspond to
    :param nominal                      1, if the feature, the new condition should correspond to, is nominal, 0
                                        otherwise
    :param num_label_indices:           The number of elements in the array `label_indices`
    :param label_indices:               A pointer to an array of type `uint32`, shape `(num_predictions)`, representing
                                        the indices of the labels for which the refined rule may predict
    :param weights:                     A pointer to an object of type `IWeightVector` that provides access to the
                                        weights of the training examples
    :param total_sum_of_weights:        The sum of the weights of all covered training examples
    :param cache_global:                A pointer to a map that maps feature indices to structs of type
                                        `IndexedFloat32Array`, storing the indices of all training examples, as well as
                                        their values for the respective feature, sorted in ascending order by the
                                        feature values
    :param cache_local:                 A pointer to a map that maps feature indices to structs of type
                                        `IndexedFloat32ArrayWrapper`, storing the indices of the training examples that
                                        are covered by the existing rule, as well as their values for the respective
                                        feature, sorted in ascending order by the feature values
    :param feature_matrix:              A pointer to an object of type `IFeatureMatrix` that provides column-wise access
                                        to the feature values of the training examples
    :param covered_statistics_mask:     An array of type `uint32`, shape `(num_statistics)` that is used to keep track
                                        of the indices of the statistics that are covered by the existing rule. It will
                                        be updated by this function
    :param covered_statistics_target:   The value that is used to mark those elements in `covered_statistics_mask` that
                                        are covered by the existing rule
    :param num_conditions:              The number of conditions in the body of the existing rule
    :param statistics:                  A pointer to an object of type `AbstractStatistics` to be used for finding the
                                        best refinement
    :param head_refinement:             A pointer to an object of type `IHeadRefinement` that should be used to find the
                                        head of the refined rule
    :param head:                        A pointer to an object of type `PredictionCandidate`, representing the head of
                                        the existing rule
    :return:                            A struct of type `Refinement`, representing the best refinement that has been
                                        found
    """
    # Obtain array that contains the indices of the training examples sorted according to the current feature...
    cdef IndexedFloat32ArrayWrapper* indexed_array_wrapper = cache_local[feature_index]
    cdef IndexedFloat32Array* indexed_array = indexed_array_wrapper.array
    cdef IndexedFloat32* indexed_values

    if indexed_array == NULL:
        indexed_array = cache_global[feature_index]
        indexed_values = indexed_array.data

        if indexed_values == NULL:
            feature_matrix.fetchSortedFeatureValues(feature_index, indexed_array)
            indexed_values = indexed_array.data

    # Filter indices, if only a subset of the contained examples is covered...
    if num_conditions > indexed_array_wrapper.numConditions:
        __filter_any_indices(indexed_array, indexed_array_wrapper, num_conditions, covered_statistics_mask,
                             covered_statistics_target)
        indexed_array = indexed_array_wrapper.array

    # Find and return the best refinement...
    cdef unique_ptr[IRuleRefinement] rule_refinement_ptr
    rule_refinement_ptr.reset(new ExactRuleRefinementImpl(statistics, indexed_array, weights, total_sum_of_weights,
                                                          feature_index, nominal))
    return rule_refinement_ptr.get().findRefinement(head_refinement, head, num_label_indices, label_indices)


# TODO Remove function
cdef inline uint32 __filter_current_indices(unordered_map[uint32, IndexedFloat32ArrayWrapper*] &cache_local,
                                            uint32 feature_index,  IndexedFloat32Array* indexed_array,
                                            intp condition_start, intp condition_end, Comparator condition_comparator,
                                            bint covered, uint32 num_conditions, uint32* covered_statistics_mask,
                                            uint32 covered_statistics_target, AbstractStatistics* statistics,
                                            IWeightVector* weights):
    """
    Filters an array that contains the indices of the examples that are covered by the previous rule, as well as their
    values for a certain feature, after a new condition that corresponds to said feature has been added, such that the
    filtered array does only contain the indices and feature values of the examples that are covered by the new rule.
    The filtered array is stored in a given struct of type `IndexedFloat32ArrayWrapper` and the given statistics are
    updated accordingly.

    :param cache_local:                 A pointer to a map that maps feature indices to structs of type
                                        `IndexedFloat32ArrayWrapper`, storing the indices of the training examples that
                                        are covered by the existing rule, as well as their values for the respective
                                        feature, sorted in ascending order by the feature values
    :param feature_index:               The index of the feature
    :param indexed_array:               A pointer to a struct of type `IndexedFloat32Array` that stores a pointer to the
                                        C-array to be filtered, as well as the number of elements in said array
    :param condition_start:             The element in `indexed_values` that corresponds to the first example
                                        (inclusive) included in the `IStatisticsSubset` that is covered by the new
                                        condition
    :param condition_end:               The element in `indexed_values` that corresponds to the last example (exclusive)
    :param condition_comparator:        The type of the operator that is used by the new condition
    :param covered                      1, if the examples in range [condition_start, condition_end) are covered by the
                                        new condition and the remaining ones are not, 0, if the examples in said range
                                        are not covered and the remaining ones are
    :param num_conditions:              The total number of conditions in the rule's body (including the new one)
    :param covered_statistics_mask:     An array of type `uint32`, shape `(num_statistics)` that is used to keep track
                                        of the indices of the statistics that are covered by the previous rule. It will
                                        be updated by this function
    :param covered_statistics_target:   The value that is used to mark those elements in `covered_statistics_mask` that
                                        are covered by the previous rule
    :param statistics:                  A pointer to an object of type `AbstractStatistics` to be notified about the
                                        examples that must be considered when searching for the next refinement, i.e.,
                                        the examples that are covered by the new rule
    :param weights:                     A pointer to an an object of type `IWeightVector` that provides access to the
                                        weights of the training examples
    :return:                            The value that is used to mark those elements in the updated
                                        `covered_statistics_mask` that are covered by the new rule
    """
    cdef IndexedFloat32* indexed_values = indexed_array.data
    cdef uint32 num_indexed_values = indexed_array.numElements
    cdef bint descending = condition_end < condition_start
    cdef uint32 updated_target, weight, index, num_steps, i, r, j
    cdef intp start, end, direction

    # Determine the number of elements in the filtered array...
    cdef uint32 num_condition_steps = abs(condition_start - condition_end)
    cdef uint32 num_elements = num_condition_steps

    if not covered:
        num_elements = (num_indexed_values - num_elements) if num_indexed_values > num_elements else 0

    # Allocate filtered array...
    cdef IndexedFloat32* filtered_array = NULL

    if num_elements > 0:
        filtered_array = <IndexedFloat32*>malloc(num_elements * sizeof(IndexedFloat32))

    if descending:
        direction = -1
        i = num_elements - 1
    else:
        direction = 1
        i = 0

    if covered:
        updated_target = num_conditions
        statistics.resetCoveredStatistics()

        # Retain the indices at positions [condition_start, condition_end) and set the corresponding values in
        # `covered_statistics_mask` to `num_conditions`, which marks them as covered (because
        # `updated_target == num_conditions`)...
        for j in range(num_condition_steps):
            r = condition_start + (j * direction)
            index = indexed_values[r].index
            covered_statistics_mask[index] = num_conditions
            filtered_array[i].index = index
            filtered_array[i].value = indexed_values[r].value
            weight = weights.getValue(index)
            statistics.updateCoveredStatistic(index, weight, False)
            i += direction
    else:
        updated_target = covered_statistics_target

        if descending:
            start = num_indexed_values - 1
            end = -1
        else:
            start = 0
            end = num_indexed_values

        if condition_comparator == Comparator.NEQ:
            # Retain the indices at positions [start, condition_start), while leaving the corresponding values in
            # `covered_statistics_mask` untouched, such that all previously covered examples in said range are still
            # marked as covered, while previously uncovered examples are still marked as uncovered...
            num_steps = abs(start - condition_start)

            for j in range(num_steps):
                r = start + (j * direction)
                filtered_array[i].index = indexed_values[r].index
                filtered_array[i].value = indexed_values[r].value
                i += direction

        # Discard the indices at positions [condition_start, condition_end) and set the corresponding values in
        # `covered_statistics_mask` to `num_conditions`, which marks them as uncovered (because
        # `updated_target != num_conditions`)...
        for j in range(num_condition_steps):
            r = condition_start + (j * direction)
            index = indexed_values[r].index
            covered_statistics_mask[index] = num_conditions
            weight = weights.getValue(index)
            statistics.updateCoveredStatistic(index, weight, True)

        # Retain the indices at positions [condition_end, end), while leaving the corresponding values in
        # `covered_statistics_mask` untouched, such that all previously covered examples in said range are still marked
        # as covered, while previously uncovered examples are still marked as uncovered...
        num_steps = abs(condition_end - end)

        for j in range(num_steps):
            r = condition_end + (j * direction)
            filtered_array[i].index = indexed_values[r].index
            filtered_array[i].value = indexed_values[r].value
            i += direction

    cdef IndexedFloat32ArrayWrapper* indexed_array_wrapper = cache_local[feature_index]
    cdef IndexedFloat32Array* filtered_indexed_array = indexed_array_wrapper.array

    if filtered_indexed_array == NULL:
        filtered_indexed_array = <IndexedFloat32Array*>malloc(sizeof(IndexedFloat32Array))
        indexed_array_wrapper.array = filtered_indexed_array
    else:
        free(filtered_indexed_array.data)

    filtered_indexed_array.data = filtered_array
    filtered_indexed_array.numElements = num_elements
    indexed_array_wrapper.numConditions = num_conditions
    return updated_target


# TODO Remove function
cdef inline void __filter_any_indices(IndexedFloat32Array* indexed_array,
                                      IndexedFloat32ArrayWrapper* indexed_array_wrapper, uint32 num_conditions,
                                      uint32* covered_statistics_mask, uint32 covered_statistics_target) nogil:
    """
    Filters an array that contains the indices of examples, as well as their values for a certain feature, such that the
    filtered array does only contain the indices and feature values of the examples that are covered by the current
    rule. The filtered array is stored in a given struct of type `IndexedFloat32ArrayWrapper`.

    :param indexed_array:               A pointer to a struct of type `IndexedFloat32Array` that stores a pointer to the
                                        C-array to be filtered, as well as the number of elements in said array
    :param indexed_array_wrapper:       A pointer to a struct of type `IndexedFloat32ArrayWrapper` that should be used
                                        to store the filtered array
    :param num_conditions:              The total number of conditions in the current rule's body
    :param covered_statistics_mask:     An array of type `uint32`, shape `(num_statistics)` that is used to keep track
                                        of the indices of the statistics that are covered by the previous rule. It will
                                        be updated by this function
    :param covered_statistics_target:   The value that is used to mark those elements in `covered_statistics_mask` that
                                        are covered by the previous rule
    """
    cdef IndexedFloat32Array* filtered_indexed_array = indexed_array_wrapper.array
    cdef IndexedFloat32* filtered_array = NULL

    if filtered_indexed_array != NULL:
        filtered_array = filtered_indexed_array.data

    cdef uint32 max_elements = indexed_array.numElements
    cdef uint32 i = 0
    cdef IndexedFloat32* indexed_values
    cdef uint32 index, r

    if max_elements > 0:
        indexed_values = indexed_array.data

        if filtered_array == NULL:
            filtered_array = <IndexedFloat32*>malloc(max_elements * sizeof(IndexedFloat32))

        for r in range(max_elements):
            index = indexed_values[r].index

            if covered_statistics_mask[index] == covered_statistics_target:
                filtered_array[i].index = index
                filtered_array[i].value = indexed_values[r].value
                i += 1

    if i == 0:
        free(filtered_array)
        filtered_array = NULL
    elif i < max_elements:
        filtered_array = <IndexedFloat32*>realloc(filtered_array, i * sizeof(IndexedFloat32))

    if filtered_indexed_array == NULL:
        filtered_indexed_array = <IndexedFloat32Array*>malloc(sizeof(IndexedFloat32Array))

    filtered_indexed_array.data = filtered_array
    filtered_indexed_array.numElements = i
    indexed_array_wrapper.array = filtered_indexed_array
    indexed_array_wrapper.numConditions = num_conditions


cdef inline Condition __make_condition(uint32 feature_index, Comparator comparator, float32 threshold):
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
