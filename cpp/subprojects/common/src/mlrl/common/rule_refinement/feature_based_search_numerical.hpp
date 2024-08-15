/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector_numerical.hpp"
#include "mlrl/common/rule_refinement/refinement.hpp"
#include "mlrl/common/statistics/statistics_subset_weighted.hpp"
#include "mlrl/common/util/math.hpp"

template<typename Comparator>
static inline void searchForNumericalRefinementInternally(const NumericalFeatureVector& featureVector,
                                                          IWeightedStatisticsSubset& statisticsSubset,
                                                          Comparator& comparator, uint32 numExamplesWithNonZeroWeights,
                                                          uint32 minCoverage, Refinement& refinement) {
    float32 sparseValue = featureVector.sparseValue;
    float32 previousValue = sparseValue;
    uint32 numFeatureValues = featureVector.numElements;
    uint32 numCovered = 0;
    int64 i = 0;

    // Traverse examples with feature values `f < sparseValue` in ascending order until the first example with non-zero
    // weight is encountered...
    for (; i < numFeatureValues; i++) {
        const IndexedValue<float32>& entry = featureVector[i];
        float32 currentValue = entry.value;

        if (!(currentValue < sparseValue)) {
            break;
        }

        uint32 index = entry.index;

        if (statisticsSubset.hasNonZeroWeight(index)) {
            // Add the example to the subset to mark it as covered by upcoming refinements...
            statisticsSubset.addToSubset(index);
            numCovered++;
            previousValue = currentValue;
            break;
        }
    }

    // Traverse the remaining examples with feature values `f < sparseValue` in ascending order...
    if (numCovered > 0) {
        for (i = i + 1; i < numFeatureValues; i++) {
            const IndexedValue<float32>& entry = featureVector[i];
            float32 currentValue = entry.value;

            if (!(currentValue < sparseValue)) {
                break;
            }

            uint32 index = entry.index;

            // Do only consider examples with non-zero weights...
            if (statisticsSubset.hasNonZeroWeight(index)) {
                // Thresholds that separate between examples with the same feature value must not be considered...
                if (!isEqual(previousValue, currentValue)) {
                    // Check if a condition using the <= operator covers at least `minCoverage` examples...
                    if (numCovered >= minCoverage) {
                        // Determine the best prediction for examples covered by a condition using the <= operator...
                        const IScoreVector& scoreVector = statisticsSubset.calculateScores();

                        // Check if the quality of the prediction is better than the quality of the current rule...
                        if (comparator.isImprovement(scoreVector)) {
                            refinement.start = 0;
                            refinement.end = i;
                            refinement.inverse = false;
                            refinement.numCovered = numCovered;
                            refinement.comparator = NUMERICAL_LEQ;
                            refinement.threshold = util::arithmeticMean(previousValue, currentValue);
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
                            refinement.comparator = NUMERICAL_GR;
                            refinement.threshold = util::arithmeticMean(previousValue, currentValue);
                            comparator.pushRefinement(refinement, scoreVector);
                        }
                    }
                }

                // Add the example to the subset to mark it as covered by upcoming refinements...
                statisticsSubset.addToSubset(index);
                numCovered++;
            }

            // Remember the feature value of the current example even if its weight is zero, because the thresholds of
            // potential conditions should be calculated based on all examples...
            previousValue = currentValue;
        }

        // Reset the subset, if any examples with feature value `f < sparseValue` have been processed...
        statisticsSubset.resetSubset();
    }

    // Traverse examples with feature values `f >= sparseValue` in descending order until the first example with
    // non-zero weight is encountered...
    float32 lastValueLessThanSparseValue = previousValue;
    int64 firstExampleWithSparseValueOrGreater = i;
    uint32 numCoveredLessThanSparseValue = numCovered;
    numCovered = 0;

    for (i = numFeatureValues - 1; i >= firstExampleWithSparseValueOrGreater; i--) {
        const IndexedValue<float32>& entry = featureVector[i];
        uint32 index = entry.index;

        if (statisticsSubset.hasNonZeroWeight(index)) {
            // Add the example to the subset to mark it as covered by upcoming refinements...
            statisticsSubset.addToSubset(index);
            numCovered++;
            previousValue = entry.value;
            break;
        }
    }

    // Traverse the remaining examples with feature values `f >= sparseValue` in descending order...
    if (numCovered > 0 && i > 0) {
        for (i = i - 1; i > firstExampleWithSparseValueOrGreater; i--) {
            const IndexedValue<float32>& entry = featureVector[i];
            float32 currentValue = entry.value;
            uint32 index = entry.index;

            // Do only consider examples with non-zero weights...
            if (statisticsSubset.hasNonZeroWeight(index)) {
                // Thresholds that separate between examples with the same feature value must not be considered...
                if (!isEqual(previousValue, currentValue)) {
                    // Check if a condition using the > operator covers at least `minCoverage` examples...
                    if (numCovered >= minCoverage) {
                        // Determine the best prediction for the covered examples...
                        const IScoreVector& scoreVector = statisticsSubset.calculateScores();

                        // Check if the quality of the prediction is better than the quality of the current rule...
                        if (comparator.isImprovement(scoreVector)) {
                            refinement.start = i + 1;
                            refinement.end = numFeatureValues;
                            refinement.inverse = false;
                            refinement.numCovered = numCovered;
                            refinement.comparator = NUMERICAL_GR;
                            refinement.threshold = util::arithmeticMean(currentValue, previousValue);
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
                            refinement.end = numFeatureValues;
                            refinement.inverse = true;
                            refinement.numCovered = numUncovered;
                            refinement.comparator = NUMERICAL_LEQ;
                            refinement.threshold = util::arithmeticMean(currentValue, previousValue);
                            comparator.pushRefinement(refinement, scoreVector);
                        }
                    }
                }

                // Add the example to the subset to mark it as covered by upcoming refinements...
                statisticsSubset.addToSubset(index);
                numCovered++;
            }

            // Remember the feature value of the current example even if its weight is zero, because the thresholds of
            // potential conditions should be calculated based on all examples...
            previousValue = currentValue;
        }
    }

    // If there are examples with sparse feature values, we must evaluate conditions separating these examples from the
    // ones that have already been traversed...
    bool sparse = featureVector.sparse;

    if (sparse) {
        // Check if the condition `f > arithmeticMean(sparseValue, previousValue)` covers at least `minCoverage`
        // examples...
        if (numCovered >= minCoverage) {
            // Determine the best prediction for examples covered by the condition...
            const IScoreVector& scoreVector = statisticsSubset.calculateScores();

            // Check if the quality of the prediction is better than the quality of the current rule...
            if (comparator.isImprovement(scoreVector)) {
                refinement.start = firstExampleWithSparseValueOrGreater;
                refinement.end = numFeatureValues;
                refinement.numCovered = numCovered;
                refinement.inverse = false;
                refinement.comparator = NUMERICAL_GR;
                refinement.threshold = util::arithmeticMean(sparseValue, previousValue);
                comparator.pushRefinement(refinement, scoreVector);
            }
        }

        // Check if the condition `f <= arithmeticMean(sparseValue, previousValue)` covers at least `minCoverage`
        // examples...
        uint32 numUncovered = numExamplesWithNonZeroWeights - numCovered;

        if (numUncovered >= minCoverage) {
            // Determine the best prediction for examples covered by the condition...
            const IScoreVector& scoreVector = statisticsSubset.calculateScoresUncovered();

            // Check if the quality of the prediction is better than the quality of the current rule...
            if (comparator.isImprovement(scoreVector)) {
                refinement.start = firstExampleWithSparseValueOrGreater;
                refinement.end = numFeatureValues;
                refinement.numCovered = numUncovered;
                refinement.inverse = true;
                refinement.comparator = NUMERICAL_LEQ;
                refinement.threshold = util::arithmeticMean(sparseValue, previousValue);
                comparator.pushRefinement(refinement, scoreVector);
            }
        }
    }

    // If there have been examples with feature values `f < sparseValue`, we must evaluate conditions that separate
    // these examples from the remaining ones...
    if (numCoveredLessThanSparseValue > 0 && numCoveredLessThanSparseValue < numExamplesWithNonZeroWeights) {
        // Check if the condition `f <= arithmeticMean(lastValueLessThanSparseValue, sparseValue)` or
        // `f <= arithmeticMean(lastValueLessThanSparseValue, previousValue)` covers at least `minCoverage` examples...
        if (numCoveredLessThanSparseValue >= minCoverage) {
            // Determine the best prediction for the examples covered by the condition...
            const IScoreVector& scoreVector = statisticsSubset.calculateScoresAccumulated();

            // Check if the quality of the prediction is better than the quality of the current rule...
            if (comparator.isImprovement(scoreVector)) {
                refinement.start = 0;
                refinement.end = firstExampleWithSparseValueOrGreater;
                refinement.numCovered = numCoveredLessThanSparseValue;
                refinement.inverse = false;
                refinement.comparator = NUMERICAL_LEQ;

                if (sparse) {
                    // If the condition separates an example with feature value `f < sparseValue` from an example with
                    // sparse feature value...
                    refinement.threshold = util::arithmeticMean(lastValueLessThanSparseValue, sparseValue);
                } else {
                    // If the condition separates an example with feature value `f < 0` from an example with feature
                    // value `f > sparseValue`...
                    refinement.threshold = util::arithmeticMean(lastValueLessThanSparseValue, previousValue);
                }

                comparator.pushRefinement(refinement, scoreVector);
            }
        }

        // Check if the condition `f > arithmeticMean(lastValueLessThanSparseValue, sparseValue)` or
        // `f > arithmeticMean(lastValueLessThanSparseValue, previousValue)` covers at least `minCoverage` examples...
        uint32 numUncovered = numExamplesWithNonZeroWeights - numCoveredLessThanSparseValue;

        if (numUncovered >= minCoverage) {
            // Determine the best prediction for the examples covered by the condition...
            const IScoreVector& scoreVector = statisticsSubset.calculateScoresUncoveredAccumulated();

            // Check if the quality of the prediction is better than the quality of the current rule...
            if (comparator.isImprovement(scoreVector)) {
                refinement.start = 0;
                refinement.end = firstExampleWithSparseValueOrGreater;
                refinement.numCovered = numUncovered;
                refinement.inverse = true;
                refinement.comparator = NUMERICAL_GR;

                if (sparse) {
                    // If the condition separates an example with feature value `f < sparseValue` from an example with
                    // sparse feature value...
                    refinement.threshold = util::arithmeticMean(lastValueLessThanSparseValue, sparseValue);
                } else {
                    // If the condition separates an example with feature value `f < sparseValue` from an example with
                    // feature value `f > sparseValue`...
                    refinement.threshold = util::arithmeticMean(lastValueLessThanSparseValue, previousValue);
                }

                comparator.pushRefinement(refinement, scoreVector);
            }
        }
    }
}
