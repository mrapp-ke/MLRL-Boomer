#include "common/rule_refinement/rule_refinement_exact.hpp"
#include "common/math/math.hpp"


static inline uint32 upperBound(FeatureVector::const_iterator iterator, uint32 start, uint32 end, float32 threshold) {
    while (start < end) {
        uint32 pivot = start + ((end - start) / 2);
        float32 featureValue = iterator[pivot].value;

        if (featureValue <= threshold) {
            start = pivot + 1;
        } else {
            end = pivot;
        }
    }

    return start;
}

static inline int64 adjustSplit(FeatureVector::const_iterator iterator, int64 conditionEnd, int64 conditionPrevious,
                                float32 threshold) {
    if (conditionEnd < conditionPrevious) {
        int64 bound = upperBound(iterator, conditionEnd + 1, conditionPrevious, threshold);
        return bound - 1;
    } else {
        return upperBound(iterator, conditionPrevious + 1, conditionEnd, threshold);
    }
}

template<typename T>
ExactRuleRefinement<T>::ExactRuleRefinement(const T& labelIndices, uint32 numExamples, uint32 featureIndex,
                                            bool nominal, bool hasZeroWeights,
                                            std::unique_ptr<IRuleRefinementCallback<FeatureVector>> callbackPtr)
    : labelIndices_(labelIndices), numExamples_(numExamples), featureIndex_(featureIndex), nominal_(nominal),
      hasZeroWeights_(hasZeroWeights), callbackPtr_(std::move(callbackPtr)) {

}

template<typename T>
void ExactRuleRefinement<T>::findRefinement(SingleRefinementComparator& comparator) {
    Refinement refinement;
    refinement.featureIndex = featureIndex_;

    // Invoke the callback...
    std::unique_ptr<IRuleRefinementCallback<FeatureVector>::Result> callbackResultPtr = callbackPtr_->get();
    const IImmutableWeightedStatistics& statistics = callbackResultPtr->statistics_;
    const FeatureVector& featureVector = callbackResultPtr->vector_;
    FeatureVector::const_iterator featureVectorIterator = featureVector.cbegin();
    uint32 numFeatureValues = featureVector.getNumElements();

    // Create a new, empty subset of the statistics...
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr = statistics.createSubset(labelIndices_);

    for (auto it = featureVector.missing_indices_cbegin(); it != featureVector.missing_indices_cend(); it++) {
        uint32 i = *it;
        statisticsSubsetPtr->addToMissing(i);
    }

    // In the following, we start by processing all examples with feature values < 0...
    uint32 numExamples = 0;
    int64 firstR = 0;
    int64 lastNegativeR = -1;
    float32 previousThreshold = 0;
    int64 previousR = 0;
    int64 r;

    // Traverse examples with feature values < 0 in ascending order until the first example with non-zero weight is
    // encountered...
    for (r = 0; r < numFeatureValues; r++) {
        float32 currentThreshold = featureVectorIterator[r].value;

        if (currentThreshold >= 0) {
            break;
        }

        lastNegativeR = r;
        uint32 i = featureVectorIterator[r].index;

        if (statisticsSubsetPtr->hasNonZeroWeight(i)) {
            // Add the example to the subset to mark it as covered by upcoming refinements...
            statisticsSubsetPtr->addToSubset(i);
            numExamples++;
            previousThreshold = currentThreshold;
            previousR = r;
            break;
        }
    }

    uint32 accumulatedNumExamples = numExamples;

    // Traverse the remaining examples with feature values < 0 in ascending order...
    if (numExamples > 0) {
        for (r = r + 1; r < numFeatureValues; r++) {
            float32 currentThreshold = featureVectorIterator[r].value;

            if (currentThreshold >= 0) {
                break;
            }

            lastNegativeR = r;
            uint32 i = featureVectorIterator[r].index;

            // Do only consider examples that are included in the current sub-sample...
            if (statisticsSubsetPtr->hasNonZeroWeight(i)) {
                // Split points between examples with the same feature value must not be considered...
                if (previousThreshold != currentThreshold) {
                    // Find and evaluate the best head for the current refinement, if a condition that uses the <=
                    // operator (or the == operator in case of a nominal feature) is used...
                    const IScoreVector& scoreVector = statisticsSubsetPtr->evaluate();

                    // If the refinement is better than the current rule...
                    if (comparator.isImprovement(scoreVector)) {
                        refinement.start = firstR;
                        refinement.end = r;
                        refinement.previous = previousR;
                        refinement.numCovered = numExamples;
                        refinement.covered = true;

                        if (nominal_) {
                            refinement.comparator = EQ;
                            refinement.threshold = previousThreshold;
                        } else {
                            refinement.comparator = LEQ;
                            refinement.threshold = arithmeticMean(previousThreshold, currentThreshold);
                        }

                        comparator.pushRefinement(refinement, scoreVector);
                    }

                    // Find and evaluate the best head for the current refinement, if a condition that uses the >
                    // operator (or the != operator in case of a nominal feature) is used...
                    const IScoreVector& scoreVector2 = statisticsSubsetPtr->evaluateUncovered();

                    // If the refinement is better than the current rule...
                    if (comparator.isImprovement(scoreVector2)) {
                        refinement.start = firstR;
                        refinement.end = r;
                        refinement.previous = previousR;
                        refinement.numCovered = (numExamples_ - numExamples);
                        refinement.covered = false;

                        if (nominal_) {
                            refinement.comparator = NEQ;
                            refinement.threshold = previousThreshold;
                        } else {
                            refinement.comparator = GR;
                            refinement.threshold = arithmeticMean(previousThreshold, currentThreshold);
                        }

                        comparator.pushRefinement(refinement, scoreVector2);
                    }

                    // Reset the subset in case of a nominal feature, as the previous examples will not be covered by
                    // the next condition...
                    if (nominal_) {
                        statisticsSubsetPtr->resetSubset();
                        numExamples = 0;
                        firstR = r;
                    }
                }

                previousThreshold = currentThreshold;
                previousR = r;

                // Add the example to the subset to mark it as covered by upcoming refinements...
                statisticsSubsetPtr->addToSubset(i);
                numExamples++;
                accumulatedNumExamples++;
            }
        }

        // If the feature is nominal and the examples that have been iterated so far do not all have the same feature
        // value, or if not all examples have been iterated so far, we must evaluate additional conditions
        // `f == previous_threshold` and `f != previous_threshold`...
        if (nominal_ && numExamples > 0 && (numExamples < accumulatedNumExamples
                                            || accumulatedNumExamples < numExamples_)) {
            // Find and evaluate the best head for the current refinement, if a condition that uses the == operator is
            // used...
            const IScoreVector& scoreVector = statisticsSubsetPtr->evaluate();

            // If the refinement is better than the current rule...
            if (comparator.isImprovement(scoreVector)) {
                refinement.start = firstR;
                refinement.end = (lastNegativeR + 1);
                refinement.previous = previousR;
                refinement.numCovered = numExamples;
                refinement.covered = true;
                refinement.comparator = EQ;
                refinement.threshold = previousThreshold;
                comparator.pushRefinement(refinement, scoreVector);
            }

            // Find and evaluate the best head for the current refinement, if a condition that uses the != operator is
            // used...
            const IScoreVector& scoreVector2 = statisticsSubsetPtr->evaluateUncovered();

            // If the refinement is better than the current rule...
            if (comparator.isImprovement(scoreVector2)) {
                refinement.start = firstR;
                refinement.end = (lastNegativeR + 1);
                refinement.previous = previousR;
                refinement.numCovered = (numExamples_ - numExamples);
                refinement.covered = false;
                refinement.comparator = NEQ;
                refinement.threshold = previousThreshold;
                comparator.pushRefinement(refinement, scoreVector2);
            }
        }

        // Reset the subset, if any examples with feature value < 0 have been processed...
        statisticsSubsetPtr->resetSubset();
    }

    float32 previousThresholdNegative = previousThreshold;
    int64 previousRNegative = previousR;
    uint32 accumulatedNumExamplesNegative = accumulatedNumExamples;

    // We continue by processing all examples with feature values >= 0...
    numExamples = 0;
    firstR = ((int64) numFeatureValues) - 1;

    // Traverse examples with feature values >= 0 in descending order until the first example with non-zero weight is
    // encountered...
    for (r = firstR; r > lastNegativeR; r--) {
        uint32 i = featureVectorIterator[r].index;

        if (statisticsSubsetPtr->hasNonZeroWeight(i)) {
            // Add the example to the subset to mark it as covered by upcoming refinements...
            statisticsSubsetPtr->addToSubset(i);
            numExamples++;
            previousThreshold = featureVectorIterator[r].value;
            previousR = r;
            break;
        }
    }

    accumulatedNumExamples = numExamples;

    // Traverse the remaining examples with feature values >= 0 in descending order...
    if (numExamples > 0) {
        for (r = r - 1; r > lastNegativeR; r--) {
            uint32 i = featureVectorIterator[r].index;

            // Do only consider examples that are included in the current sub-sample...
            if (statisticsSubsetPtr->hasNonZeroWeight(i)) {
                float32 currentThreshold = featureVectorIterator[r].value;

                // Split points between examples with the same feature value must not be considered...
                if (previousThreshold != currentThreshold) {
                    // Find and evaluate the best head for the current refinement, if a condition that uses the
                    // > operator (or the == operator in case of a nominal feature) is used...
                    const IScoreVector& scoreVector = statisticsSubsetPtr->evaluate();

                    // If the refinement is better than the current rule...
                    if (comparator.isImprovement(scoreVector)) {
                        refinement.start = firstR;
                        refinement.end = r;
                        refinement.previous = previousR;
                        refinement.numCovered = numExamples;
                        refinement.covered = true;

                        if (nominal_) {
                            refinement.comparator = EQ;
                            refinement.threshold = previousThreshold;
                        } else {
                            refinement.comparator = GR;
                            refinement.threshold = arithmeticMean(currentThreshold, previousThreshold);
                        }

                        comparator.pushRefinement(refinement, scoreVector);
                    }

                    // Find and evaluate the best head for the current refinement, if a condition that uses the <=
                    // operator (or the != operator in case of a nominal feature) is used...
                    const IScoreVector& scoreVector2 = statisticsSubsetPtr->evaluateUncovered();

                    // If the refinement is better than the current rule...
                    if (comparator.isImprovement(scoreVector2)) {
                        refinement.start = firstR;
                        refinement.end = r;
                        refinement.previous = previousR;
                        refinement.numCovered = (numExamples_ - numExamples);
                        refinement.covered = false;

                        if (nominal_) {
                            refinement.comparator = NEQ;
                            refinement.threshold = previousThreshold;
                        } else {
                            refinement.comparator = LEQ;
                            refinement.threshold = arithmeticMean(currentThreshold, previousThreshold);
                        }

                        comparator.pushRefinement(refinement, scoreVector2);
                    }

                    // Reset the subset in case of a nominal feature, as the previous examples will not be covered by
                    // the next condition...
                    if (nominal_) {
                        statisticsSubsetPtr->resetSubset();
                        numExamples = 0;
                        firstR = r;
                    }
                }

                previousThreshold = currentThreshold;
                previousR = r;

                // Add the example to the subset to mark it as covered by upcoming refinements...
                statisticsSubsetPtr->addToSubset(i);
                numExamples++;
                accumulatedNumExamples++;
            }
        }
    }

    // If the feature is nominal and the examples with feature values >= 0 that have been iterated so far do not all
    // have the same feature value, we must evaluate additional conditions `f == previous_threshold` and
    // `f != previous_threshold`...
    if (nominal_ && numExamples > 0 && numExamples < accumulatedNumExamples) {
        // Find and evaluate the best head for the current refinement, if a condition that uses the == operator is
        // used...
        const IScoreVector& scoreVector = statisticsSubsetPtr->evaluate();

        // If the refinement is better than the current rule...
        if (comparator.isImprovement(scoreVector)) {
            refinement.start = firstR;
            refinement.end = lastNegativeR;
            refinement.previous = previousR;
            refinement.numCovered = numExamples;
            refinement.covered = true;
            refinement.comparator = EQ;
            refinement.threshold = previousThreshold;
            comparator.pushRefinement(refinement, scoreVector);
        }

        // Find and evaluate the best head for the current refinement, if a condition that uses the != operator is
        // used...
        const IScoreVector& scoreVector2 = statisticsSubsetPtr->evaluateUncovered();

        // If the refinement is better than the current rule...
        if (comparator.isImprovement(scoreVector2)) {
            refinement.start = firstR;
            refinement.end = lastNegativeR;
            refinement.previous = previousR;
            refinement.numCovered = (numExamples_ - numExamples);
            refinement.covered = false;
            refinement.comparator = NEQ;
            refinement.threshold = previousThreshold;
            comparator.pushRefinement(refinement, scoreVector2);
        }
    }

    uint32 totalAccumulatedNumExamples = accumulatedNumExamplesNegative + accumulatedNumExamples;

    // If the number of all examples that have been iterated so far (including those with feature values < 0 and those
    // with feature values >= 0) is less than the total number of examples, this means that there are examples with
    // sparse, i.e. zero, feature values. In such case, we must explicitly test conditions that separate these examples
    // from the ones that have already been iterated...
    if (totalAccumulatedNumExamples > 0 && totalAccumulatedNumExamples < numExamples_) {
        // If the feature is nominal, we must reset the subset once again to ensure that the accumulated state includes
        // all examples that have been processed so far...
        if (nominal_) {
            statisticsSubsetPtr->resetSubset();
            firstR = ((int64) numFeatureValues) - 1;
        }

        // Find and evaluate the best head for the current refinement, if the condition `f > previous_threshold / 2` (or
        // the condition `f != 0` in case of a nominal feature) is used...
        const IScoreVector& scoreVector =
            nominal_? statisticsSubsetPtr->evaluateAccumulated() : statisticsSubsetPtr->evaluate();

        // If the refinement is better than the current rule...
        if (comparator.isImprovement(scoreVector)) {
            refinement.start = firstR;
            refinement.covered = true;

            if (nominal_) {
                refinement.end = -1;
                refinement.previous = -1;
                refinement.numCovered = totalAccumulatedNumExamples;
                refinement.comparator = NEQ;
                refinement.threshold = 0.0;
            } else {
                refinement.end = lastNegativeR;
                refinement.previous = previousR;
                refinement.numCovered = accumulatedNumExamples;
                refinement.comparator = GR;
                refinement.threshold = previousThreshold * 0.5;
            }

            comparator.pushRefinement(refinement, scoreVector);
        }

        // Find and evaluate the best head for the current refinement, if the condition `f <= previous_threshold / 2`
        // (or `f == 0` in case of a nominal feature) is used...
        const IScoreVector& scoreVector2 =
            nominal_ ? statisticsSubsetPtr->evaluateUncoveredAccumulated() : statisticsSubsetPtr->evaluateUncovered();

        // If the refinement is better than the current rule...
        if (comparator.isImprovement(scoreVector2)) {
            refinement.start = firstR;
            refinement.covered = false;

            if (nominal_) {
                refinement.end = -1;
                refinement.previous = -1;
                refinement.numCovered = (numExamples_ - totalAccumulatedNumExamples);
                refinement.comparator = EQ;
                refinement.threshold = 0.0;
            } else {
                refinement.end = lastNegativeR;
                refinement.previous = previousR;
                refinement.numCovered = (numExamples_ - accumulatedNumExamples);
                refinement.comparator = LEQ;
                refinement.threshold = previousThreshold * 0.5;
            }

            comparator.pushRefinement(refinement, scoreVector2);
        }
    }

    // If the feature is numerical and there are other examples than those with feature values < 0 that have been
    // processed earlier, we must evaluate additional conditions that separate the examples with feature values < 0 from
    // the remaining ones (unlike in the nominal case, these conditions cannot be evaluated earlier, because it remains
    // unclear what the thresholds of the conditions should be until the examples with feature values >= 0 have been
    // processed).
    if (!nominal_ && accumulatedNumExamplesNegative > 0 && accumulatedNumExamplesNegative < numExamples_) {
        // Find and evaluate the best head for the current refinement, if the condition that uses the <= operator is
        // used...
        const IScoreVector& scoreVector = statisticsSubsetPtr->evaluateAccumulated();

        // If the refinement is better than the current rule...
        if (comparator.isImprovement(scoreVector)) {
            refinement.start = 0;
            refinement.end = (lastNegativeR + 1);
            refinement.previous = previousRNegative;
            refinement.numCovered = accumulatedNumExamplesNegative;
            refinement.covered = true;
            refinement.comparator = LEQ;

            if (totalAccumulatedNumExamples < numExamples_) {
                // If the condition separates an example with feature value < 0 from an (sparse) example with feature
                // value == 0
                refinement.threshold = previousThresholdNegative * 0.5;
            } else {
                // If the condition separates an example with feature value < 0 from an example with feature value > 0
                refinement.threshold = arithmeticMean(previousThresholdNegative, previousThreshold);
            }

            comparator.pushRefinement(refinement, scoreVector);
        }

        // Find and evaluate the best head for the current refinement, if the condition that uses the > operator is
        // used...
        const IScoreVector& scoreVector2 = statisticsSubsetPtr->evaluateUncoveredAccumulated();

        // If the refinement is better than the current rule...
        if (comparator.isImprovement(scoreVector2)) {
            refinement.start = 0;
            refinement.end = (lastNegativeR + 1);
            refinement.previous = previousRNegative;
            refinement.numCovered = (numExamples_ - accumulatedNumExamplesNegative);
            refinement.covered = false;
            refinement.comparator = GR;

            if (totalAccumulatedNumExamples < numExamples_) {
                // If the condition separates an example with feature value < 0 from an (sparse) example with feature
                // value == 0
                refinement.threshold = previousThresholdNegative * 0.5;
            } else {
                // If the condition separates an example with feature value < 0 from an example with feature value > 0
                refinement.threshold = arithmeticMean(previousThresholdNegative, previousThreshold);
            }

            comparator.pushRefinement(refinement, scoreVector2);
        }
    }

    // If there are examples with zero weights, those examples have not been considered when searching for potential
    // refinements. In this case, we need to identify the examples that are covered by a refinement, including those
    // that have previously been ignored, and adjust the value `refinement.end`, which specifies the position that
    // separates the covered from the uncovered examples, accordingly.
    if (hasZeroWeights_) {
        for (auto it = comparator.begin(); it != comparator.end(); it++) {
            Refinement& ref = *it;

            if (std::abs(ref.previous - ref.end) > 1) {
                ref.end = adjustSplit(featureVectorIterator, ref.end, ref.previous, ref.threshold);
            }
        }
    }
}

template class ExactRuleRefinement<CompleteIndexVector>;
template class ExactRuleRefinement<PartialIndexVector>;
