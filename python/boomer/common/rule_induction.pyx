"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement algorithms for inducing individual classification rules.
"""
from boomer.common._arrays cimport float64, array_uint32
from boomer.common._tuples cimport IndexedFloat32, IndexedFloat32ArrayWrapper
from boomer.common._predictions cimport Prediction, PredictionCandidate
from boomer.common.rules cimport Condition, Comparator
from boomer.common.rule_refinement cimport Refinement, IRuleRefinement
from boomer.common.statistics cimport AbstractStatistics, IStatisticsSubset
from boomer.common.sub_sampling cimport IWeightVector, IIndexVector
from boomer.common.thresholds cimport IThresholdsSubset, ExactThresholdsImpl, ThresholdsSubsetImpl

from libc.math cimport fabs
from libc.stdlib cimport abs, malloc, realloc, free

from libcpp.unordered_map cimport unordered_map
from libcpp.list cimport list as double_linked_list
from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr

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
        cdef uint32 num_examples = thresholds.getNumRows()
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
        cdef uint32 num_covered_examples, num_sampled_features, weight, f
        cdef intp c

        # Sub-sample examples...
        cdef unique_ptr[IWeightVector] weights_ptr
        weights_ptr.reset(instance_sub_sampling.subSample(num_examples, rng))

        # Create a new subset of the given thresholds...
        cdef unique_ptr[IThresholdsSubset] thresholds_subset_ptr
        thresholds_subset_ptr.reset(thresholds.createSubset(weights_ptr.get()))

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
                    current_refinement = current_rule_refinement.findRefinement(head_refinement, best_refinement.head,
                                                                                num_predictions, label_indices)
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
                    conditions.push_back(__create_condition(best_refinement.featureIndex, best_refinement.comparator,
                                                            best_refinement.threshold))
                    num_conditions += 1
                    num_conditions_per_comparator[<uint32>best_refinement.comparator] += 1

                    if max_head_refinements > 0 and num_conditions >= max_head_refinements:
                        # Keep the labels for which the rule predicts, if the head should not be further refined...
                        num_predictions = best_refinement.head.numPredictions_
                        label_indices = best_refinement.head.labelIndices_

                    # Filter the current subset of thresholds by applying the best refinement that has been found...
                    thresholds_subset_ptr.get().applyRefinement(best_refinement)
                    num_covered_examples = best_refinement.coveredWeights

                    if num_covered_examples <= min_coverage:
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


cdef inline Condition __create_condition(uint32 feature_index, Comparator comparator, float32 threshold):
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
