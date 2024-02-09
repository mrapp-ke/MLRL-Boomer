/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector_binary.hpp"
#include "mlrl/common/rule_refinement/refinement.hpp"
#include "rule_refinement_search_nominal_common.hpp"

template<typename Comparator>
static inline void searchForBinaryRefinementInternally(const BinaryFeatureVector& featureVector,
                                                       IWeightedStatisticsSubset& statisticsSubset,
                                                       Comparator& comparator, uint32 numExamplesWithNonZeroWeights,
                                                       uint32 minCoverage, Refinement& refinement) {
    // Mark all examples corresponding to the minority value as covered...
    uint32 numCovered = addAllToSubset(statisticsSubset, featureVector, 0);

    // Check if a condition covering all examples corresponding to the minority value covers at least `minCoverage`
    // examples...
    if (numCovered >= minCoverage) {
        // Determine the best prediction for the examples corresponding to the minority value...
        const IScoreVector& scoreVector = statisticsSubset.calculateScores();

        // Check if the quality of the prediction is better than the quality of the current rule...
        if (comparator.isImprovement(scoreVector)) {
            refinement.start = 0;
            refinement.end = 1;
            refinement.inverse = false;
            refinement.numCovered = numCovered;
            refinement.comparator = NOMINAL_EQ;
            refinement.threshold = featureVector.values_cbegin()[0];
            comparator.pushRefinement(refinement, scoreVector);
        }
    }

    // Check if a condition covering all examples corresponding to the majority value covers at least `minCoverage`
    // examples...
    uint32 numUncovered = numExamplesWithNonZeroWeights - numCovered;

    if (numUncovered >= minCoverage) {
        // Determine the best prediction for the examples corresponding to the majority value...
        const IScoreVector& scoreVector = statisticsSubset.calculateScoresUncovered();

        // Check if the quality of the prediction is better than the quality of the current rule...
        if (comparator.isImprovement(scoreVector)) {
            refinement.start = 0;
            refinement.end = 1;
            refinement.inverse = true;
            refinement.numCovered = numUncovered;
            refinement.comparator = NOMINAL_EQ;
            refinement.threshold = featureVector.majorityValue;
            comparator.pushRefinement(refinement, scoreVector);
        }
    }
}
