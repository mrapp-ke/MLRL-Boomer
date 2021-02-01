#include "rule_induction_top_down.h"
#include "../indices/index_vector_full.h"
#include <unordered_map>


void TopDownRuleInduction::induceDefaultRule(IStatisticsProvider& statisticsProvider,
                                             const IHeadRefinementFactory* headRefinementFactory,
                                             IModelBuilder& modelBuilder) const {
    if (headRefinementFactory != nullptr) {
        IStatistics& statistics = statisticsProvider.get();
        uint32 numStatistics = statistics.getNumStatistics();
        uint32 numLabels = statistics.getNumLabels();
        statistics.resetSampledStatistics();

        for (uint32 i = 0; i < numStatistics; i++) {
            statistics.addSampledStatistic(i, 1);
        }

        FullIndexVector labelIndices(numLabels);
        std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = labelIndices.createSubset(statistics);
        std::unique_ptr<IHeadRefinement> headRefinementPtr = headRefinementFactory->create(labelIndices);
        headRefinementPtr->findHead(nullptr, *statisticsSubsetPtr, true, false);
        std::unique_ptr<AbstractEvaluatedPrediction> defaultPredictionPtr = headRefinementPtr->pollHead();
        statisticsProvider.switchRuleEvaluation();

        for (uint32 i = 0; i < numStatistics; i++) {
            defaultPredictionPtr->apply(statistics, i);
        }

        modelBuilder.setDefaultRule(*defaultPredictionPtr);
    } else {
        statisticsProvider.switchRuleEvaluation();
    }
}

bool TopDownRuleInduction::induceRule(IThresholds& thresholds, const IIndexVector& labelIndices,
                                      const IWeightVector& weights, const IFeatureSubSampling& featureSubSampling,
                                      const IPruning& pruning, const IPostProcessor& postProcessor, uint32 minCoverage,
                                      intp maxConditions, intp maxHeadRefinements, int numThreads, RNG& rng,
                                      IModelBuilder& modelBuilder) const {
    // The total number of features
    uint32 numFeatures = thresholds.getNumFeatures();
    // True, if the rule is learned on a sub-sample of the available training examples, False otherwise
    bool instanceSubSamplingUsed = weights.hasZeroWeights();
    // The label indices for which the next refinement of the rule may predict
    const IIndexVector* currentLabelIndices = &labelIndices;
    // A (stack-allocated) list that contains the conditions in the rule's body (in the order they have been learned)
    ConditionList conditions;
    // The total number of conditions
    uint32 numConditions = 0;
    // A map that stores a pointer to an object of type `IRuleRefinement` for each feature
    std::unordered_map<uint32, IRuleRefinement*> ruleRefinements; // TODO Should we really use a pointer
    // An unique pointer to the best refinement of the current rule
    std::unique_ptr<Refinement> bestRefinementPtr = std::make_unique<Refinement>();
    // Whether a refinement of the current rule has been found
    bool foundRefinement = true;

    // Temporary variables
    uint32 numCoveredExamples;

    // Create a new subset of the given thresholds...
    std::unique_ptr<IThresholdsSubset> thresholdsSubsetPtr = thresholds.createSubset(weights);

    // Search for the best refinement until no improvement in terms of the rule's quality score is possible anymore or
    // the maximum number of conditions has been reached...
    while (foundRefinement && (maxConditions == -1 || numConditions < maxConditions)) {
        foundRefinement = false;

        // Sample features...
        std::unique_ptr<IIndexVector> sampledFeatureIndicesPtr = featureSubSampling.subSample(numFeatures, rng);
        uint32 numSampledFeatures = sampledFeatureIndicesPtr->getNumElements();

        // For each feature, create an object of type `IRuleRefinement`...
        for (intp i = 0; i < numSampledFeatures; i++) {
            uint32 featureIndex = sampledFeatureIndicesPtr->getIndex((uint32) i);
            std::unique_ptr<IRuleRefinement> ruleRefinementPtr = currentLabelIndices->createRuleRefinement(
                *thresholdsSubsetPtr, featureIndex);
            ruleRefinements[featureIndex] = ruleRefinementPtr.release();
        }

        // Search for the best condition among all available features to be added to the current rule...
        for (intp i = 0; i < numSampledFeatures; i++) {  // TODO Use OpenMP
            uint32 featureIndex = sampledFeatureIndicesPtr->getIndex((uint32) i);
            IRuleRefinement* ruleRefinement = ruleRefinements[featureIndex];
            ruleRefinement->findRefinement(bestRefinementPtr->headPtr.get());
        }

        // Pick the best refinement among the refinements that have been found for the different features...
        for (intp i = 0; i < numSampledFeatures; i++) {
            uint32 featureIndex = sampledFeatureIndicesPtr->getIndex((uint32) i);
            IRuleRefinement* ruleRefinement = ruleRefinements[featureIndex];
            std::unique_ptr<Refinement> refinementPtr = ruleRefinement->pollRefinement();

            if (refinementPtr->isBetterThan(*bestRefinementPtr)) {
                bestRefinementPtr = std::move(refinementPtr);
                foundRefinement = true;
            }

            delete ruleRefinement;
        }

        if (foundRefinement) {
            // Filter the current subset of thresholds by applying the best refinement that has been found...
            thresholdsSubsetPtr->filterThresholds(*bestRefinementPtr);
            numCoveredExamples = bestRefinementPtr->coveredWeights;

            // Add the new condition...
            conditions.addCondition(*bestRefinementPtr);
            numConditions++;

            // Keep the labels for which the rule predicts, if the head should not be further refined...
            if (maxHeadRefinements > 0 && numConditions >= maxHeadRefinements) {
                currentLabelIndices = bestRefinementPtr->headPtr.get();
            }

            // Abort refinement process if the rule is not allowed to cover less examples...
            if (numCoveredExamples <= minCoverage) {
                break;
            }
        }
    }

    if (bestRefinementPtr->headPtr.get() == nullptr) {
        // No rule could be induced, because no useful condition could be found. This might be the case, if all examples
        // have the same values for the considered features.
        return false;
    } else {
        if (instanceSubSamplingUsed) {
            // Prune rule...
            std::unique_ptr<CoverageMask> coverageMaskPtr = pruning.prune(*thresholdsSubsetPtr, conditions,
                                                                          *bestRefinementPtr->headPtr);

            // Re-calculate the scores in the head based on the entire training data...
            const CoverageMask& coverageMask =
                coverageMaskPtr.get() != nullptr ? *coverageMaskPtr : thresholdsSubsetPtr->getCoverageMask();
            thresholdsSubsetPtr->recalculatePrediction(coverageMask, *bestRefinementPtr);
        }

        // Apply post-processor...
        postProcessor.postProcess(*bestRefinementPtr->headPtr);

        // Update the statistics by applying the predictions of the new rule...
        thresholdsSubsetPtr->applyPrediction(*bestRefinementPtr->headPtr);

        // Add the induced rule to the model...
        modelBuilder.addRule(conditions, *bestRefinementPtr->headPtr);
        return true;
    }
}
