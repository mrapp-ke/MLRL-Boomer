"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement algorithms for inducing individual classification rules.
"""
from boomer.common._types cimport float32
from boomer.common._indices cimport IIndexVector, FullIndexVector
from boomer.common.head_refinement cimport IHeadRefinement, AbstractEvaluatedPrediction
from boomer.common.rules cimport Condition, Comparator, ConditionList
from boomer.common.rule_refinement cimport Refinement, IRuleRefinement
from boomer.common.statistics cimport IStatistics, IStatisticsSubset
from boomer.common.sampling cimport IWeightVector
from boomer.common.thresholds cimport IThresholdsSubset, CoverageMask

from libcpp.unordered_map cimport unordered_map
from libcpp.memory cimport unique_ptr, make_unique
from libcpp.utility cimport move

from cython.operator cimport dereference
from cython.parallel cimport prange


cdef class RuleInduction:
    """
    A base class for all classes that implement an algorithm for the induction of individual classification rules.
    """

    cdef void induce_default_rule(self, StatisticsProvider statistics_provider,
                                  IHeadRefinementFactory* head_refinement_factory, ModelBuilder model_builder):
        """
        Induces the default rule.

        :param statistics_provider:     A `StatisticsProvider` that provides access to the statistics which should serve
                                        as the basis for inducing the default rule
        :param head_refinement_factory: A pointer to an object of type `IHeadRefinementFactory` that allows to create
                                        instances of the class that should be used to find the head of the default rule
                                        or a null pointer, if no default rule should be induced
        :param model_builder:           The builder, the default rule should be added to
        """
        pass

    cdef bint induce_rule(self, IThresholds* thresholds, INominalFeatureMask* nominal_feature_mask,
                          IFeatureMatrix* feature_matrix, ILabelSubSampling* label_sub_sampling,
                          IInstanceSubSampling* instance_sub_sampling, IFeatureSubSampling* feature_sub_sampling,
                          IPruning* pruning, IPostProcessor* post_processor, uint32 min_coverage, intp max_conditions,
                          intp max_head_refinements, int num_threads, RNG* rng, ModelBuilder model_builder):
        """
        Induces a new classification rule.

        :param thresholds:              A pointer to an object of type `IThresholds` that provides access to the
                                        thresholds that may be used by the conditions of rules
        :param nominal_feature_mask:    A pointer to an object of type `INominalFeatureMask` that provides access to the
                                        information whether individual features are nominal or not
        :param feature_matrix:          A pointer to an object of type `IFeatureMatrix` that provides column-wise access
                                        to the feature values of the training examples
        :param label_sub_sampling:      A pointer to an object of type `ILabelSubSampling`, implementing the strategy
                                        that should be used to sub-sample the labels
        :param instance_sub_sampling:   A pointer to an object of type `IInstanceSubSampling`, implementing the strategy
                                        that should be used to sub-sample the training examples
        :param feature_sub_sampling:    A pointer to an object of type `IFeatureSubSampling`, implementing the strategy
                                        that should be used to sub-sample the available features
        :param pruning:                 A pointer to an object of type `IPruning`, implementing the strategy that should
                                        be used to prune the rule
        :param post_processor:          A pointer to an object of type `IPostProcessor`, implementing the post-processor
                                        that should be used to post-process the rules once they have been learned
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

    cdef void induce_default_rule(self, StatisticsProvider statistics_provider,
                                  IHeadRefinementFactory* head_refinement_factory, ModelBuilder model_builder):
        cdef unique_ptr[IHeadRefinement] head_refinement_ptr
        cdef unique_ptr[AbstractEvaluatedPrediction] default_prediction_ptr
        cdef unique_ptr[IStatisticsSubset] statistics_subset_ptr
        cdef unique_ptr[FullIndexVector] label_indices_ptr
        cdef IStatistics* statistics
        cdef uint32 num_statistics, num_labels, i

        if head_refinement_factory != NULL:
            statistics = statistics_provider.get()
            num_statistics = statistics.getNumStatistics()
            num_labels = statistics.getNumLabels()
            label_indices_ptr = make_unique[FullIndexVector](num_labels)
            statistics.resetSampledStatistics()

            for i in range(num_statistics):
                statistics.addSampledStatistic(i, 1)

            statistics_subset_ptr = label_indices_ptr.get().createSubset(dereference(statistics))
            head_refinement_ptr = head_refinement_factory.create(dereference(label_indices_ptr.get()))
            head_refinement_ptr.get().findHead(NULL, dereference(statistics_subset_ptr.get()), True, False)
            default_prediction_ptr = head_refinement_ptr.get().pollHead()
            statistics_provider.switch_rule_evaluation()

            for i in range(num_statistics):
                default_prediction_ptr.get().apply(dereference(statistics), i)

            model_builder.set_default_rule(default_prediction_ptr.get())
        else:
            statistics_provider.switch_rule_evaluation()

    cdef bint induce_rule(self, IThresholds* thresholds, INominalFeatureMask* nominal_feature_mask,
                          IFeatureMatrix* feature_matrix, ILabelSubSampling* label_sub_sampling,
                          IInstanceSubSampling* instance_sub_sampling, IFeatureSubSampling* feature_sub_sampling,
                          IPruning* pruning, IPostProcessor* post_processor, uint32 min_coverage, intp max_conditions,
                          intp max_head_refinements, int num_threads, RNG* rng, ModelBuilder model_builder):
        # The total number of statistics
        cdef uint32 num_examples = thresholds.getNumExamples()
        # The total number of features
        cdef uint32 num_features = thresholds.getNumFeatures()
        # The total number of labels
        cdef uint32 num_labels = thresholds.getNumLabels()
        # A (stack-allocated) list that contains the conditions in the rule's body (in the order they have been learned)
        cdef ConditionList conditions
        # The total number of conditions
        cdef uint32 num_conditions = 0
        # A map that stores a pointer to an object of type `IRuleRefinement` for each feature
        cdef unordered_map[uint32, IRuleRefinement*] rule_refinements  # Stack-allocated map
        # An unique pointer to the best refinement of the current rule
        cdef unique_ptr[Refinement] best_refinement_ptr = make_unique[Refinement]()
        # Whether a refinement of the current rule has been found
        cdef bint found_refinement = True

        # Temporary variables
        cdef unique_ptr[IRuleRefinement] rule_refinement_ptr
        cdef unique_ptr[Refinement] current_refinement_ptr
        cdef IRuleRefinement* rule_refinement
        cdef unique_ptr[IIndexVector] sampled_feature_indices_ptr
        cdef uint32 num_covered_examples, num_sampled_features, weight, f
        cdef unique_ptr[CoverageMask] coverage_mask_ptr
        cdef const CoverageMask* coverage_mask
        cdef intp c

        # Sub-sample examples...
        cdef unique_ptr[IWeightVector] weights_ptr = instance_sub_sampling.subSample(num_examples, dereference(rng))
        cdef bint instance_sub_sampling_used = weights_ptr.get().hasZeroWeights()

        # Sub-sample labels...
        cdef unique_ptr[IIndexVector] label_indices_ptr = label_sub_sampling.subSample(num_labels, dereference(rng))
        cdef IIndexVector* label_indices = label_indices_ptr.get()

        # Create a new subset of the given thresholds...
        cdef unique_ptr[IThresholdsSubset] thresholds_subset_ptr = thresholds.createSubset(
            dereference(weights_ptr.get()))

        # Search for the best refinement until no improvement in terms of the rule's quality score is possible anymore
        # or the maximum number of conditions has been reached...
        while found_refinement and (max_conditions == -1 or num_conditions < max_conditions):
            found_refinement = False

            # Sub-sample features...
            sampled_feature_indices_ptr = feature_sub_sampling.subSample(num_features, dereference(rng))
            num_sampled_features = sampled_feature_indices_ptr.get().getNumElements()

            # For each feature, create an object of type `IRuleRefinement`...
            for c in range(num_sampled_features):
                f = sampled_feature_indices_ptr.get().getIndex(<uint32>c)
                rule_refinement_ptr = label_indices.createRuleRefinement(dereference(thresholds_subset_ptr.get()), f)
                rule_refinements[f] = rule_refinement_ptr.release()

            # Search for the best condition among all available features to be added to the current rule...
            for c in prange(num_sampled_features, nogil=True, schedule='dynamic', num_threads=num_threads):
                f = sampled_feature_indices_ptr.get().getIndex(<uint32>c)
                rule_refinement = rule_refinements[f]
                rule_refinement.findRefinement(best_refinement_ptr.get().headPtr.get())

            # Pick the best refinement among the refinements that have been found for the different features...
            for c in range(num_sampled_features):
                f = sampled_feature_indices_ptr.get().getIndex(<uint32>c)
                rule_refinement = rule_refinements[f]
                current_refinement_ptr = move(rule_refinement.pollRefinement())

                if current_refinement_ptr.get().isBetterThan(dereference(best_refinement_ptr.get())):
                    best_refinement_ptr = move(current_refinement_ptr)
                    found_refinement = True

                del rule_refinement

            if found_refinement:
                # Filter the current subset of thresholds by applying the best refinement that has been found...
                thresholds_subset_ptr.get().filterThresholds(dereference(best_refinement_ptr.get()))
                num_covered_examples = best_refinement_ptr.get().coveredWeights

                # Add the new condition...
                conditions.append(__create_condition(best_refinement_ptr.get()))
                num_conditions += 1

                if max_head_refinements > 0 and num_conditions >= max_head_refinements:
                    # Keep the labels for which the rule predicts, if the head should not be further refined...
                    label_indices = <IIndexVector*>best_refinement_ptr.get().headPtr.get()

                if num_covered_examples <= min_coverage:
                    # Abort refinement process if the rule is not allowed to cover less examples...
                    break

        if best_refinement_ptr.get().headPtr.get() == NULL:
            # No rule could be induced, because no useful condition could be found. This might be the case, if all
            # examples have the same values for the considered features.
            return False
        else:
            if instance_sub_sampling_used:
                # Prune rule...
                coverage_mask_ptr = pruning.prune(dereference(thresholds_subset_ptr.get()), conditions,
                                                  dereference(best_refinement_ptr.get().headPtr.get()))
                coverage_mask = coverage_mask_ptr.get()

                # Re-calculate the scores in the head based on the entire training data...
                if coverage_mask == NULL:
                    coverage_mask = &thresholds_subset_ptr.get().getCoverageMask()

                thresholds_subset_ptr.get().recalculatePrediction(dereference(coverage_mask),
                                                                  dereference(best_refinement_ptr.get()))

            # Apply post-processor...
            post_processor.postProcess(dereference(best_refinement_ptr.get().headPtr.get()))

            # Update the statistics by applying the predictions of the new rule...
            thresholds_subset_ptr.get().applyPrediction(dereference(best_refinement_ptr.get().headPtr.get()))

            # Add the induced rule to the model...
            model_builder.add_rule(best_refinement_ptr.get().headPtr.get(), conditions)
            return True


cdef inline Condition __create_condition(Refinement* refinement):
    """
    Creates and returns a new condition from a specific refinement.

    :param refinement: A pointer to an object of type `Refinement`
    """
    cdef Condition condition
    condition.featureIndex = refinement.featureIndex
    condition.comparator = refinement.comparator
    condition.threshold = refinement.threshold
    condition.start = refinement.start
    condition.end = refinement.end
    condition.covered = refinement.covered
    condition.coveredWeights = refinement.coveredWeights
    return condition
