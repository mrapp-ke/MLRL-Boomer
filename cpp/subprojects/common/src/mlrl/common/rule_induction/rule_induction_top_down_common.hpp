/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/rule_refinement/feature_subspace.hpp"
#include "mlrl/common/util/openmp.hpp"

#include <memory>

/**
 * Stores an unique pointer to an object of type `IRuleRefinement` that may be used to search for potential refinements
 * of a rule, as well as to an object of template type `RefinementComparator` that allows comparing different
 * refinements and keeping track of the best one(s).
 *
 * @tparam The type of the comparator that allows comparing different refinements and keeping track of the best one(s)
 */
template<typename RefinementComparator>
struct RuleRefinementEntry final {
    public:

        /**
         * An unique pointer to an object of type `IRuleRefinement` that may be used to search for potential refinements
         * of a rule.
         */
        std::unique_ptr<IRuleRefinement> ruleRefinementPtr;

        /**
         * An unique pointer to an object of template type `RefinementComparator` that allows comparing different
         * refinements and keeping track of the best one(s).
         */
        std::unique_ptr<RefinementComparator> comparatorPtr;
};

/**
 * Finds the best refinement(s) of an existing rule across multiple features.
 *
 * @tparam RefinementComparator     The type of the comparator that is used to compare the potential refinements
 * @param refinementComparator      A reference to an object of template type `RefinementComparator` that should be used
 *                                  to compare the potential refinements
 * @param featureSubspace           A reference to an object of type `IFeatureSubspace` that should be used to search
 *                                  for the potential refinements
 * @param featureIndices            A reference to an object of type `IIndexVector` that provides access to the indices
 *                                  of the features that should be considered
 * @param outputIndices             A reference to an object of type `IIndexVector` that provides access to the indices
 *                                  of the outputs for which the refinement(s) may predict
 * @param minCoverage               The minimum number of training examples that must be covered by potential
 *                                  refinements
 * @param multiThreadingSettings    An object of type `MultiThreadingSettings` that stores the settings to be used for
 *                                  searching for potential refinements across multiple features in parallel
 * @return                          True, if at least one refinement has been found, false otherwise
 */
template<typename RefinementComparator>
static inline bool findRefinement(RefinementComparator& refinementComparator, IFeatureSubspace& featureSubspace,
                                  const IIndexVector& featureIndices, const IIndexVector& outputIndices,
                                  uint32 minCoverage, MultiThreadingSettings multiThreadingSettings) {
    bool foundRefinement = false;

    // For each feature, create an object of type `RuleRefinement<RefinementComparator>`...
    uint32 numFeatures = featureIndices.getNumElements();
    RuleRefinementEntry<RefinementComparator>* ruleRefinementEntries =
      new RuleRefinementEntry<RefinementComparator>[numFeatures];

    for (uint32 i = 0; i < numFeatures; i++) {
        uint32 featureIndex = featureIndices.getIndex(i);
        RuleRefinementEntry<RefinementComparator>& ruleRefinementEntry = ruleRefinementEntries[i];
        ruleRefinementEntry.comparatorPtr = std::make_unique<RefinementComparator>(refinementComparator);
        ruleRefinementEntry.ruleRefinementPtr = outputIndices.createRuleRefinement(featureSubspace, featureIndex);
    }

    // Search for the best condition among all available features to be added to the current rule...
#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numFeatures) firstprivate(ruleRefinementEntries) firstprivate(minCoverage) \
      schedule(dynamic) num_threads(multiThreadingSettings.numThreads)
#endif
    for (int64 i = 0; i < numFeatures; i++) {
        RuleRefinementEntry<RefinementComparator>& ruleRefinementEntry = ruleRefinementEntries[i];
        ruleRefinementEntry.ruleRefinementPtr->findRefinement(*ruleRefinementEntry.comparatorPtr, minCoverage);
    }

    // Pick the best refinement among the refinements that have been found for the different features...
    for (uint32 i = 0; i < numFeatures; i++) {
        RuleRefinementEntry<RefinementComparator>& ruleRefinementEntry = ruleRefinementEntries[i];
        foundRefinement |= refinementComparator.merge(*ruleRefinementEntry.comparatorPtr);
    }

    delete[] ruleRefinementEntries;
    return foundRefinement;
}
