/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_based_search_binned_common.hpp"
#include "mlrl/common/input/feature_vector_binary.hpp"
#include "mlrl/common/rule_refinement/refinement.hpp"
#include "mlrl/common/statistics/statistics_subset_resettable.hpp"

template<typename Comparator>
static inline void searchForBinaryRefinementInternally(const BinaryFeatureVector& featureVector,
                                                       IResettableStatisticsSubset& statisticsSubset,
                                                       Comparator& comparator, uint32 numExamplesWithNonZeroWeights,
                                                       uint32 minCoverage, Refinement& refinement) {
    // Mark all examples corresponding to the minority value as covered...
    uint32 numCovered = addAllToSubset(statisticsSubset, featureVector, 0);

    // Check if a condition covering all examples corresponding to the minority value covers at least `minCoverage`
    // examples...
    if (numCovered >= minCoverage) {
        // Determine the best prediction for the examples corresponding to the minority value...
        std::unique_ptr<StatisticsUpdateCandidate> updateCandidatePtr = statisticsSubset.calculateScores();

        // Check if the quality of the prediction is better than the quality of the current rule...
        if (comparator.isImprovement(*updateCandidatePtr)) {
            refinement.start = 0;
            refinement.end = 1;
            refinement.inverse = false;
            refinement.numCovered = numCovered;
            refinement.comparator = NOMINAL_EQ;
            refinement.threshold = featureVector.values_cbegin()[0];
            comparator.pushRefinement(refinement, *updateCandidatePtr);
        }
    }

    // Check if a condition covering all examples corresponding to the majority value covers at least `minCoverage`
    // examples...
    uint32 numUncovered = numExamplesWithNonZeroWeights - numCovered;

    if (numUncovered >= minCoverage) {
        // Determine the best prediction for the examples corresponding to the majority value...
        std::unique_ptr<StatisticsUpdateCandidate> updateCandidatePtr = statisticsSubset.calculateScoresUncovered();

        // Check if the quality of the prediction is better than the quality of the current rule...
        if (comparator.isImprovement(*updateCandidatePtr)) {
            refinement.start = 0;
            refinement.end = 1;
            refinement.inverse = true;
            refinement.numCovered = numUncovered;
            refinement.comparator = NOMINAL_EQ;
            refinement.threshold = featureVector.majorityValue;
            comparator.pushRefinement(refinement, *updateCandidatePtr);
        }
    }
}
