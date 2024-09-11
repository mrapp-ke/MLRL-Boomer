/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_based_search_binned_common.hpp"
#include "mlrl/common/input/feature_vector_ordinal.hpp"
#include "mlrl/common/rule_refinement/refinement.hpp"

template<typename Comparator>
static inline void searchForOrdinalRefinementInternally(const OrdinalFeatureVector& featureVector,
                                                        IResettableStatisticsSubset& statisticsSubset,
                                                        Comparator& comparator, uint32 numExamplesWithNonZeroWeights,
                                                        uint32 minCoverage, Refinement& refinement) {
    // Mark all examples corresponding to the first ordinal feature value `f < majorityValue` as covered...
    NominalFeatureVector::value_const_iterator valueIterator = featureVector.values_cbegin();
    uint32 numValues = featureVector.numValues;
    int32 majorityValue = featureVector.majorityValue;
    uint32 numCovered = 0;
    int64 i = 0;
    int32 previousValue = valueIterator[i];

    if (previousValue < majorityValue) {
        numCovered += addAllToSubset(statisticsSubset, featureVector, i);
    }

    // Traverse ordinal feature values `f < sparseValue` in ascending order...
    if (numCovered > 0) {
        for (i = i + 1; i < numValues; i++) {
            int32 currentValue = valueIterator[i];

            if (currentValue >= majorityValue) {
                break;
            }

            // Check if a condition using the <= operator covers at least `minCoverage` examples...
            if (numCovered >= minCoverage) {
                // Determine the best prediction for the examples covered by a condition using the <= operator...
                const IScoreVector& scoreVector = statisticsSubset.calculateScores();

                // Check if the quality of the prediction is better than the quality of the current rule...
                if (comparator.isImprovement(scoreVector)) {
                    refinement.start = 0;
                    refinement.end = i;
                    refinement.inverse = false;
                    refinement.numCovered = numCovered;
                    refinement.comparator = ORDINAL_LEQ;
                    refinement.threshold = previousValue;
                    comparator.pushRefinement(refinement, scoreVector);
                }
            }

            // Check if a condition using the > operator covers at least `minCoverage` examples...
            uint32 numUncovered = numExamplesWithNonZeroWeights - numCovered;

            if (numUncovered >= minCoverage) {
                // Determine the best prediction for examples covered by a condition using the > operator...
                const IScoreVector& scoreVector = statisticsSubset.calculateScoresUncovered();

                // Check if the quality of the prediction is better than the quality of the current rule...
                if (comparator.isImprovement(scoreVector)) {
                    refinement.start = 0;
                    refinement.end = i;
                    refinement.inverse = true;
                    refinement.numCovered = numUncovered;
                    refinement.comparator = ORDINAL_GR;
                    refinement.threshold = previousValue;
                    comparator.pushRefinement(refinement, scoreVector);
                }
            }

            // Mark all examples corresponding to the current ordinal feature value as covered...
            numCovered += addAllToSubset(statisticsSubset, featureVector, i);
            previousValue = currentValue;
        }

        // Reset the subset, if any examples with feature value `f < majorityValue` have been processed...
        statisticsSubset.resetSubset();
    }

    // Mark all examples corresponding to the last ordinal feature value `f > majorityValue` as covered...
    int32 lastValueLessThanMajorityValue = previousValue;
    int64 firstValueGreaterThanMajorityValue = i;
    uint32 numCoveredLessThanMajorityValue = numCovered;
    numCovered = 0;
    i = numValues - 1;

    if (valueIterator[i] > majorityValue) {
        numCovered += addAllToSubset(statisticsSubset, featureVector, i);
    }

    // Traverse ordinal feature values `f > majorityValue` in descending order...
    if (numCovered > 0) {
        for (i = i - 1; i >= firstValueGreaterThanMajorityValue; i--) {
            int32 currentValue = valueIterator[i];

            // Check if a condition using the > operator covers at least `minCoverage` examples...
            if (numCovered >= minCoverage) {
                // Determine the best prediction for the covered examples...
                const IScoreVector& scoreVector = statisticsSubset.calculateScores();

                // Check if the quality of the prediction is better than the quality of the current rule...
                if (comparator.isImprovement(scoreVector)) {
                    refinement.start = i + 1;
                    refinement.end = numValues;
                    refinement.inverse = false;
                    refinement.numCovered = numCovered;
                    refinement.comparator = ORDINAL_GR;
                    refinement.threshold = currentValue;
                    comparator.pushRefinement(refinement, scoreVector);
                }
            }

            // Check if a condition using the <= operator covers at least `minCoverage` examples...
            uint32 numUncovered = numExamplesWithNonZeroWeights - numCovered;

            if (numUncovered >= minCoverage) {
                // Determine the best prediction for the covered examples...
                const IScoreVector& scoreVector = statisticsSubset.calculateScoresUncovered();

                // Check if the quality of the prediction is better than the quality of the current rule...
                if (comparator.isImprovement(scoreVector)) {
                    refinement.start = i + 1;
                    refinement.end = numValues;
                    refinement.inverse = true;
                    refinement.numCovered = numUncovered;
                    refinement.comparator = ORDINAL_LEQ;
                    refinement.threshold = currentValue;
                    comparator.pushRefinement(refinement, scoreVector);
                }
            }

            // Mark all examples corresponding to the current ordinal feature value as covered...
            numCovered += addAllToSubset(statisticsSubset, featureVector, i);
            previousValue = currentValue;
        }
    }

    // Check if the condition `f > majorityValue` covers at least `minCoverage` examples...
    if (numCovered >= minCoverage) {
        // Determine the best prediction for examples covered by the condition...
        const IScoreVector& scoreVector = statisticsSubset.calculateScores();

        // Check if the quality of the prediction is better than the quality of the current rule...
        if (comparator.isImprovement(scoreVector)) {
            refinement.start = firstValueGreaterThanMajorityValue;
            refinement.end = numValues;
            refinement.numCovered = numCovered;
            refinement.inverse = false;
            refinement.comparator = ORDINAL_GR;
            refinement.threshold = majorityValue;
            comparator.pushRefinement(refinement, scoreVector);
        }
    }

    // Check if the condition `f <= majorityValue` covers at least `minCoverage` examples...
    uint32 numUncovered = numExamplesWithNonZeroWeights - numCovered;

    if (numUncovered >= minCoverage) {
        // Determine the best prediction for examples covered by the condition...
        const IScoreVector& scoreVector = statisticsSubset.calculateScoresUncovered();

        // Check if the quality of the prediction is better than the quality of the current rule...
        if (comparator.isImprovement(scoreVector)) {
            refinement.start = firstValueGreaterThanMajorityValue;
            refinement.end = numValues;
            refinement.numCovered = numUncovered;
            refinement.inverse = true;
            refinement.comparator = ORDINAL_LEQ;
            refinement.threshold = majorityValue;
            comparator.pushRefinement(refinement, scoreVector);
        }
    }

    // If there have been examples with feature values `f < majorityValue`, we must evaluate conditions that separate
    // these examples from the remaining ones...
    if (numCoveredLessThanMajorityValue > 0 && numCoveredLessThanMajorityValue < numExamplesWithNonZeroWeights) {
        // Check if the condition `f <= lastValueLessThanMajorityValue` covers at least `minCoverage` examples...
        if (numCoveredLessThanMajorityValue >= minCoverage) {
            // Determine the best prediction for the examples covered by the condition...
            const IScoreVector& scoreVector = statisticsSubset.calculateScoresAccumulated();

            // Check if the quality of the prediction is better than the quality of the current rule...
            if (comparator.isImprovement(scoreVector)) {
                refinement.start = 0;
                refinement.end = firstValueGreaterThanMajorityValue;
                refinement.numCovered = numCoveredLessThanMajorityValue;
                refinement.inverse = false;
                refinement.comparator = ORDINAL_LEQ;
                refinement.threshold = lastValueLessThanMajorityValue;
                comparator.pushRefinement(refinement, scoreVector);
            }
        }

        // Check if the condition `f > lastValueLessThanMajorityValue` covers at least `minCoverage` examples...
        numUncovered = numExamplesWithNonZeroWeights - numCoveredLessThanMajorityValue;

        if (numUncovered >= minCoverage) {
            // Determine the best prediction for the examples covered by the condition...
            const IScoreVector& scoreVector = statisticsSubset.calculateScoresUncoveredAccumulated();

            // Check if the quality of the prediction is better than the quality of the current rule...
            if (comparator.isImprovement(scoreVector)) {
                refinement.start = 0;
                refinement.end = firstValueGreaterThanMajorityValue;
                refinement.numCovered = numUncovered;
                refinement.inverse = true;
                refinement.comparator = ORDINAL_GR;
                refinement.threshold = lastValueLessThanMajorityValue;
                comparator.pushRefinement(refinement, scoreVector);
            }
        }
    }
}
