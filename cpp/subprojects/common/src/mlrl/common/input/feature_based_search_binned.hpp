/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_based_search_binned_common.hpp"
#include "mlrl/common/input/feature_vector_binned.hpp"
#include "mlrl/common/rule_refinement/refinement.hpp"
#include "mlrl/common/statistics/statistics_subset_resettable.hpp"

template<typename Comparator>
static inline void searchForBinnedRefinementInternally(const BinnedFeatureVector& featureVector,
                                                       IResettableStatisticsSubset& statisticsSubset,
                                                       Comparator& comparator, uint32 numExamplesWithNonZeroWeights,
                                                       uint32 minCoverage, Refinement& refinement) {
    // Mark all examples corresponding to the first bin with index `i < sparseBinIndex` as covered...
    BinnedFeatureVector::threshold_const_iterator thresholdIterator = featureVector.thresholds_cbegin();
    uint32 numBins = featureVector.numBins;
    int32 sparseBinIndex = featureVector.sparseBinIndex;
    uint32 numCovered = 0;
    int64 i = 0;

    if (i < sparseBinIndex) {
        numCovered += addAllToSubset(statisticsSubset, featureVector, i);
    }

    // Traverse bins with indices `i < sparseBinIndex` in ascending order...
    if (numCovered > 0) {
        for (i = i + 1; i < sparseBinIndex; i++) {
            // Check if a condition using the <= operator covers at least `minCoverage` examples...
            if (numCovered >= minCoverage) {
                // Determine the best prediction for the examples covered by a condition using the <= operator...
                std::unique_ptr<StatisticsUpdateCandidate> updateCandidatePtr = statisticsSubset.calculateScores();

                // Check if the quality of the prediction is better than the quality of the current rule...
                if (comparator.isImprovement(*updateCandidatePtr)) {
                    refinement.start = 0;
                    refinement.end = i;
                    refinement.inverse = false;
                    refinement.numCovered = numCovered;
                    refinement.comparator = NUMERICAL_LEQ;
                    refinement.threshold = thresholdIterator[i - 1];
                    comparator.pushRefinement(refinement, *updateCandidatePtr);
                }
            }

            // Check if a condition using the > operator covers at least `minCoverage` examples...
            uint32 numUncovered = numExamplesWithNonZeroWeights - numCovered;

            if (numUncovered >= minCoverage) {
                // Determine the best prediction for examples covered by a condition using the > operator...
                std::unique_ptr<StatisticsUpdateCandidate> updateCandidatePtr =
                  statisticsSubset.calculateScoresUncovered();

                // Check if the quality of the prediction is better than the quality of the current rule...
                if (comparator.isImprovement(*updateCandidatePtr)) {
                    refinement.start = 0;
                    refinement.end = i;
                    refinement.inverse = true;
                    refinement.numCovered = numUncovered;
                    refinement.comparator = NUMERICAL_GR;
                    refinement.threshold = thresholdIterator[i - 1];
                    comparator.pushRefinement(refinement, *updateCandidatePtr);
                }
            }

            // Mark all examples corresponding to the current bin as covered...
            numCovered += addAllToSubset(statisticsSubset, featureVector, i);
        }

        // Reset the subset, if any bins with indices `i < sparseBinIndex` have been processed...
        statisticsSubset.resetSubset();
    }

    // Mark all examples corresponding to the last bin with index `i > sparseBinIndex` as covered...
    uint32 numCoveredLessThanSparseBinIndex = numCovered;
    numCovered = 0;
    i = numBins - 1;

    if (i > sparseBinIndex) {
        numCovered += addAllToSubset(statisticsSubset, featureVector, i);
    }

    // Traverse bin with indices `i > sparseBinIndex` in descending order...
    if (numCovered > 0) {
        for (i = i - 1; i > sparseBinIndex; i--) {
            // Check if a condition using the > operator covers at least `minCoverage` examples...
            if (numCovered >= minCoverage) {
                // Determine the best prediction for the covered examples...
                std::unique_ptr<StatisticsUpdateCandidate> updateCandidatePtr = statisticsSubset.calculateScores();

                // Check if the quality of the prediction is better than the quality of the current rule...
                if (comparator.isImprovement(*updateCandidatePtr)) {
                    refinement.start = i + 1;
                    refinement.end = numBins;
                    refinement.inverse = false;
                    refinement.numCovered = numCovered;
                    refinement.comparator = NUMERICAL_GR;
                    refinement.threshold = thresholdIterator[i];
                    comparator.pushRefinement(refinement, *updateCandidatePtr);
                }
            }

            // Check if a condition using the <= operator covers at least `minCoverage` examples...
            uint32 numUncovered = numExamplesWithNonZeroWeights - numCovered;

            if (numUncovered >= minCoverage) {
                // Determine the best prediction for the covered examples...
                std::unique_ptr<StatisticsUpdateCandidate> updateCandidatePtr =
                  statisticsSubset.calculateScoresUncovered();

                // Check if the quality of the prediction is better than the quality of the current rule...
                if (comparator.isImprovement(*updateCandidatePtr)) {
                    refinement.start = i + 1;
                    refinement.end = numBins;
                    refinement.inverse = true;
                    refinement.numCovered = numUncovered;
                    refinement.comparator = NUMERICAL_LEQ;
                    refinement.threshold = thresholdIterator[i];
                    comparator.pushRefinement(refinement, *updateCandidatePtr);
                }
            }

            // Mark all examples corresponding to the current bin as covered...
            numCovered += addAllToSubset(statisticsSubset, featureVector, i);
        }
    }

    // Check if a condition that covers all bins with indices `i > sparseBinIndex` covers at least `minCoverage`
    // examples...
    if (numCovered >= minCoverage) {
        // Determine the best prediction for examples covered by the condition...
        std::unique_ptr<StatisticsUpdateCandidate> updateCandidatePtr = statisticsSubset.calculateScores();

        // Check if the quality of the prediction is better than the quality of the current rule...
        if (comparator.isImprovement(*updateCandidatePtr)) {
            refinement.start = sparseBinIndex + 1;
            refinement.end = numBins;
            refinement.numCovered = numCovered;
            refinement.inverse = false;
            refinement.comparator = NUMERICAL_GR;
            refinement.threshold = thresholdIterator[sparseBinIndex];
            comparator.pushRefinement(refinement, *updateCandidatePtr);
        }
    }

    // Check if a condition that covers all bins with indices `i <= sparseBinIndex` covers at least `minCoverage`
    // examples...
    uint32 numUncovered = numExamplesWithNonZeroWeights - numCovered;

    if (numUncovered >= minCoverage) {
        // Determine the best prediction for examples covered by the condition...
        std::unique_ptr<StatisticsUpdateCandidate> updateCandidatePtr = statisticsSubset.calculateScores();

        // Check if the quality of the prediction is better than the quality of the current rule...
        if (comparator.isImprovement(*updateCandidatePtr)) {
            refinement.start = sparseBinIndex + 1;
            refinement.end = numBins;
            refinement.numCovered = numUncovered;
            refinement.inverse = true;
            refinement.comparator = NUMERICAL_LEQ;
            refinement.threshold = thresholdIterator[sparseBinIndex];
            comparator.pushRefinement(refinement, *updateCandidatePtr);
        }
    }

    // If there have been bin with indices `i < sparseBinIndex`, we must evaluate conditions that separate the examples
    // corresponding to these bins from the remaining ones...
    if (numCoveredLessThanSparseBinIndex > 0 && numCoveredLessThanSparseBinIndex < numExamplesWithNonZeroWeights) {
        // Check if a condition that covers all bins with indices `i < sparseBinIndex` covers at least `minCoverage`
        // examples...
        if (numCoveredLessThanSparseBinIndex >= minCoverage) {
            // Determine the best prediction for the examples covered by the condition...
            std::unique_ptr<StatisticsUpdateCandidate> updateCandidatePtr =
              statisticsSubset.calculateScoresAccumulated();

            // Check if the quality of the prediction is better than the quality of the current rule...
            if (comparator.isImprovement(*updateCandidatePtr)) {
                refinement.start = 0;
                refinement.end = sparseBinIndex;
                refinement.numCovered = numCoveredLessThanSparseBinIndex;
                refinement.inverse = false;
                refinement.comparator = NUMERICAL_LEQ;
                refinement.threshold = thresholdIterator[sparseBinIndex - 1];
                comparator.pushRefinement(refinement, *updateCandidatePtr);
            }
        }

        // Check if a condition that covers all bins with indices `i >= sparseBinIndex` covers at least `minCoverage`
        // examples...
        numUncovered = numExamplesWithNonZeroWeights - numCoveredLessThanSparseBinIndex;

        if (numUncovered >= minCoverage) {
            // Determine the best prediction for the examples covered by the condition...
            std::unique_ptr<StatisticsUpdateCandidate> updateCandidatePtr =
              statisticsSubset.calculateScoresUncoveredAccumulated();

            // Check if the quality of the prediction is better than the quality of the current rule...
            if (comparator.isImprovement(*updateCandidatePtr)) {
                refinement.start = 0;
                refinement.end = sparseBinIndex;
                refinement.numCovered = numUncovered;
                refinement.inverse = true;
                refinement.comparator = NUMERICAL_GR;
                refinement.threshold = thresholdIterator[sparseBinIndex - 1];
                comparator.pushRefinement(refinement, *updateCandidatePtr);
            }
        }
    }
}
