/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_based_search_binned_common.hpp"
#include "mlrl/common/input/feature_vector_nominal.hpp"
#include "mlrl/common/rule_refinement/refinement.hpp"
#include "mlrl/common/statistics/statistics_subset_resettable.hpp"

template<typename Comparator>
static inline void searchForNominalRefinementInternally(const NominalFeatureVector& featureVector,
                                                        IResettableStatisticsSubset& statisticsSubset,
                                                        Comparator& comparator, uint32 numExamplesWithNonZeroWeights,
                                                        uint32 minCoverage, Refinement& refinement) {
    NominalFeatureVector::value_const_iterator valueIterator = featureVector.values_cbegin();
    uint32 numValues = featureVector.numBins;
    uint32 numExamplesWithMinorityValue = 0;

    for (uint32 i = 0; i < numValues; i++) {
        // Mark all examples corresponding to the current minority value as covered...
        uint32 numCovered = addAllToSubset(statisticsSubset, featureVector, i);

        // Check if a condition using the == operator covers at least `minCoverage` examples...
        if (numCovered >= minCoverage) {
            // Determine the best prediction for the examples covered by a condition using the == operator...
            const IScoreVector& scoreVector = statisticsSubset.calculateScores();

            // Check if the quality of the prediction is better than the quality of the current rule...
            if (comparator.isImprovement(scoreVector)) {
                refinement.start = i;
                refinement.end = i + 1;
                refinement.inverse = false;
                refinement.numCovered = numCovered;
                refinement.comparator = NOMINAL_EQ;
                refinement.threshold = valueIterator[i];
                comparator.pushRefinement(refinement, scoreVector);
            }
        }

        // Check if a condition using the != operator covers at least `minCoverage` examples...
        uint32 numUncovered = numExamplesWithNonZeroWeights - numCovered;

        if (numUncovered >= minCoverage) {
            // Determine the best prediction for the examples covered by a condition using the != operator...
            const IScoreVector& scoreVector = statisticsSubset.calculateScoresUncovered();

            // Check if the quality of the prediction is better than the quality of the current rule...
            if (comparator.isImprovement(scoreVector)) {
                refinement.start = i;
                refinement.end = i + 1;
                refinement.inverse = true;
                refinement.numCovered = numUncovered;
                refinement.comparator = NOMINAL_NEQ;
                refinement.threshold = valueIterator[i];
                comparator.pushRefinement(refinement, scoreVector);
            }
        }

        // Reset the subset, as the previous examples will not be covered by the next condition...
        statisticsSubset.resetSubset();

        // Increment the number of examples corresponding to one of the minority values...
        numExamplesWithMinorityValue += numCovered;
    }

    // Check if a condition covering all examples corresponding to one of the minority values covers at least
    // `minCoverage` examples...
    if (numExamplesWithMinorityValue >= minCoverage) {
        // Determine the best prediction for the examples corresponding to one of the minority values...
        const IScoreVector& scoreVector = statisticsSubset.calculateScoresAccumulated();

        // Check if the quality of the prediction is better than the quality of the current rule...
        if (comparator.isImprovement(scoreVector)) {
            refinement.start = 0;
            refinement.end = numValues;
            refinement.inverse = false;
            refinement.numCovered = numExamplesWithMinorityValue;
            refinement.comparator = NOMINAL_NEQ;
            refinement.threshold = featureVector.majorityValue;
            comparator.pushRefinement(refinement, scoreVector);
        }
    }

    // Check if a condition covering all examples corresponding to the majority value covers at least `minCoverage`
    // examples...
    uint32 numExamplesWithMajorityValue = numExamplesWithNonZeroWeights - numExamplesWithMinorityValue;

    if (numExamplesWithMajorityValue >= minCoverage) {
        // Determine the best prediction for the examples corresponding to the majority value...
        const IScoreVector& scoreVector = statisticsSubset.calculateScoresUncoveredAccumulated();

        // Check if the quality of the prediction is better than the quality of the current rule...
        if (comparator.isImprovement(scoreVector)) {
            refinement.start = 0;
            refinement.end = numValues;
            refinement.inverse = true;
            refinement.numCovered = numExamplesWithMajorityValue;
            refinement.comparator = NOMINAL_EQ;
            refinement.threshold = featureVector.majorityValue;
            comparator.pushRefinement(refinement, scoreVector);
        }
    }
}
