/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector_binary.hpp"
#include "mlrl/common/rule_refinement/refinement.hpp"
#include "mlrl/common/statistics/statistics_subset_weighted.hpp"

template<typename Comparator>
static inline void searchForBinaryRefinementInternally(const BinaryFeatureVector& featureVector,
                                                       IWeightedStatisticsSubset& statisticsSubset,
                                                       Comparator& comparator, uint32 minCoverage,
                                                       Refinement& refinement) {
    BinaryFeatureVector::value_const_iterator valueIterator = featureVector.values_cbegin();
    BinaryFeatureVector::index_const_iterator indexIterator = featureVector.indices_cbegin(0);
    BinaryFeatureVector::index_const_iterator indicesEnd = featureVector.indices_cend(0);
    uint32 numIndices = indicesEnd - indexIterator;
    uint32 numCovered = 0;

    // Mark all examples corresponding to the minority value as covered...
    for (uint32 i = 0; i < numIndices; i++) {
        uint32 index = indexIterator[i];

        // Do only consider examples with non-zero weights...
        if (statisticsSubset.hasNonZeroWeight(index)) {
            statisticsSubset.addToSubset(index);
            numCovered++;
        }
    }

    // Check if a condition covering all examples corresponding to the minority value covers at least `minCoverage`
    // examples...
    if (numCovered >= minCoverage) {
        // Determine the best prediction for the examples corresponding to the minority value...
        const IScoreVector& scoreVector = statisticsSubset.calculateScores();

        // Check if the quality of the prediction is better than the quality of the current rule...
        if (comparator.isImprovement(scoreVector)) {
            refinement.start = 0;
            refinement.end = 1;
            refinement.numCovered = numCovered;
            refinement.covered = true;
            refinement.comparator = NOMINAL_EQ;
            refinement.threshold = valueIterator[0];
            comparator.pushRefinement(refinement, scoreVector);
        }
    }

    // Check if a condition covering all examples corresponding to the majority value covers at least `minCoverage`
    // examples...
    uint32 numExamplesWithNonZeroWeights = statisticsSubset.getNumNonZeroWeights();
    uint32 numUncovered = numExamplesWithNonZeroWeights - numCovered;

    if (numUncovered >= minCoverage) {
        // Determine the best prediction for the examples corresponding to the majority value...
        const IScoreVector& scoreVector = statisticsSubset.calculateScoresUncovered();

        // Check if the quality of the prediction is better than the quality of the current rule...
        if (comparator.isImprovement(scoreVector)) {
            refinement.start = 0;
            refinement.end = 1;
            refinement.numCovered = numUncovered;
            refinement.covered = false;
            refinement.comparator = NOMINAL_EQ;
            refinement.threshold = featureVector.majorityValue;
            comparator.pushRefinement(refinement, scoreVector);
        }
    }
}
