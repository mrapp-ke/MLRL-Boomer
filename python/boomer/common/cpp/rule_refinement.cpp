#include "rule_refinement.h"
#include <memory>

AbstractRuleRefinement::~AbstractRuleRefinement() {

}

Refinement AbstractRuleRefinement::findRefinement(AbstractHeadRefinement* headRefinement,
                                                  PredictionCandidate* currentHead, uint32 numLabelIndices,
                                                  const uint32* labelIndices) {
    Refinement refinement;
    return refinement;
}

RuleRefinementImpl::RuleRefinementImpl(AbstractStatistics* statistics, IndexedFloat32Array* indexedArray,
                                       const uint32* weights, uint32 totalSumOfWeights, uint32 featureIndex,
                                       bool nominal) {
    statistics_ = statistics;
    indexedArray_ = indexedArray;
    weights_ = weights;
    totalSumOfWeights_ = totalSumOfWeights;
    featureIndex_ = featureIndex;
    nominal_ = nominal;
}

RuleRefinementImpl::~RuleRefinementImpl() {

}

Refinement RuleRefinementImpl::findRefinement(AbstractHeadRefinement* headRefinement, PredictionCandidate* currentHead,
                                              uint32 numLabelIndices, const uint32* labelIndices) {
    // The current refinement of the existing rule
    Refinement refinement;
    refinement.featureIndex = featureIndex_;
    refinement.head = NULL;
    // The best head seen so far
    PredictionCandidate* bestHead = currentHead;
    // The `AbstractRefinementSearch` to be used for evaluating refinements
    std::unique_ptr<AbstractRefinementSearch> refinementSearchPtr;
    refinementSearchPtr.reset(statistics_->beginSearch(numLabelIndices, labelIndices));
    // The example indices and feature values to be iterated
    IndexedFloat32* indexedValues = indexedArray_->data;
    uint32 numIndexedValues = indexedArray_->numElements;

    // In the following, we start by processing all examples with feature values < 0...
    uint32 sumOfWeights = 0;
    intp firstR = 0;
    intp lastNegativeR = -1;
    float32 previousThreshold;
    intp r, previousR;

    // Traverse examples with feature values < 0 in ascending order until the first example with weight > 0 is
    // encountered...
    for (r = 0; r < numIndexedValues; r++) {
        float32 currentThreshold = indexedValues[r].value;

        if (currentThreshold >= 0) {
            break;
        }

        lastNegativeR = r;
        uint32 i = indexedValues[r].index;
        uint32 weight = weights_ == NULL ? 1 : weights_[i];

        if (weight > 0) {
            // Tell the search that the example will be covered by upcoming refinements...
            refinementSearchPtr.get()->updateSearch(i, weight);
            sumOfWeights += weight;
            previousThreshold = currentThreshold;
            previousR = r;
            break;
        }
    }

    uint32 accumulatedSumOfWeights = sumOfWeights;

    // Traverse the remaining examples with feature values < 0 in ascending order...
    if (sumOfWeights > 0) {
        for (r = r + 1; r < numIndexedValues; r++) {
            float32 currentThreshold = indexedValues[r].value;

            if (currentThreshold >= 0) {
                break;
            }

            lastNegativeR = r;
            uint32 i = indexedValues[r].index;
            uint32 weight = weights_ == NULL ? 1 : weights_[i];

            // Do only consider examples that are included in the current sub-sample...
            if (weight > 0) {
                // Split points between examples with the same feature value must not be considered...
                if (previousThreshold != currentThreshold) {
                    // TODO

                    if (nominal_) {
                        refinementSearchPtr.get()->resetSearch();
                        sumOfWeights = 0;
                        firstR = r;
                    }
                }

                previousThreshold = currentThreshold;
                previousR = r;

                // Tell the search that the example will be covered by upcoming refinements...
                refinementSearchPtr.get()->updateSearch(i, weight);
                sumOfWeights += weight;
                accumulatedSumOfWeights += weight;
            }
        }

        // If the feature is nominal and the examples that have been iterated so far do not all have the same feature
        // value, or if not all examples have been iterated so far, we must evaluate additional conditions
        // `f == previous_threshold` and `f != previous_threshold`...
        if (nominal_ && sumOfWeights > 0 && (sumOfWeights < accumulatedSumOfWeights
                                             || accumulatedSumOfWeights < totalSumOfWeights_)) {
            // TODO
        }

        // Reset the search, if any examples with feature value < 0 have been processed...
        refinementSearchPtr.get()->resetSearch();
    }

    float32 previousThresholdNegative = previousThreshold;
    intp previousRNegative = previousR;
    uint32 accumulatedSumOfWeightsNegative = accumulatedSumOfWeights;

    // We continue by processing all examples with feature values >= 0...
    sumOfWeights = 0;
    firstR = ((intp) numIndexedValues) - 1;

    // Traverse examples with feature values >= 0 in descending order until the first example with weight > 0 is
    // encountered...
    for (r = firstR; r > lastNegativeR; r--) {
        uint32 i = indexedValues[r].index;
        uint32 weight = weights_ == NULL ? 1 : weights_[i];

        if (weight > 0) {
            // Tell the search that the example will be covered by upcoming refinements...
            refinementSearchPtr.get()->updateSearch(i, weight);
            sumOfWeights += weight;
            previousThreshold = indexedValues[r].value;
            previousR = r;
            break;
        }
    }

    accumulatedSumOfWeights = sumOfWeights;

    // Traverse the remaining examples with feature values >= 0 in descending order...
    if (sumOfWeights > 0) {
        for (r = r - 1; r > lastNegativeR; r--) {
            uint32 i = indexedValues[r].index;
            uint32 weight = weights_ == NULL ? 1 : weights_[i];

            // Do only consider examples that are included in the current sub-sample...
            if (weight > 0) {
                float32 currentThreshold = indexedValues[r].value;

                // Split points between examples with the same feature value must not be considered...
                if (previousThreshold != currentThreshold) {
                    // TODO

                    // Reset the search in case of a nominal feature, as the previous examples will not be covered by
                    // the next condition...
                    if (nominal_) {
                        refinementSearchPtr.get()->resetSearch();
                        sumOfWeights = 0;
                        firstR = r;
                    }
                }

                previousThreshold = currentThreshold;
                previousR = r;

                // Tell the search that the example will be covered by upcoming refinements...
                refinementSearchPtr.get()->updateSearch(i, weight);
                sumOfWeights += weight;
                accumulatedSumOfWeights += weight;
            }
        }
    }

    // If the feature is nominal and the examples with feature values >= 0 that have been iterated so far do not all
    // have the same feature value, we must evaluate additional conditions `f == previous_threshold` and
    // `f != previous_threshold`...
    if (nominal_ && sumOfWeights > 0 && sumOfWeights < accumulatedSumOfWeights) {
        // TODO
    }

    uint32 totalAccumulatedSumOfWeights = accumulatedSumOfWeightsNegative + accumulatedSumOfWeights;

    // If the sum of weights of all examples that have been iterated so far (including those with feature values < 0 and
    // those with feature values >= 0) is less than the sum of of weights of all examples, this means that there are
    // examples with sparse, i.e. zero, feature values. In such case, we must explicitly test conditions that separate
    // these examples from the ones that have already been iterated...
    if (totalAccumulatedSumOfWeights > 0 && totalAccumulatedSumOfWeights < totalSumOfWeights_) {
        // If the feature is nominal, we must reset the search once again to ensure that the accumulated state includes
        // all examples that have been processed so far...
        if (nominal_) {
            refinementSearchPtr.get()->resetSearch();
            firstR = ((intp) numIndexedValues) - 1;
        }

        // TODO
    }

    // If the feature is numerical and there are other examples than those with feature values < 0 that have been
    // processed earlier, we must evaluate additional conditions that separate the examples with feature values < 0 from
    // the remaining ones (unlike in the nominal case, these conditions cannot be evaluated earlier, because it remains
    // unclear what the thresholds of the conditions should be until the examples with feature values >= 0 have been
    // processed).
    if (!nominal_ && accumulatedSumOfWeightsNegative > 0 && accumulatedSumOfWeightsNegative < totalSumOfWeights_) {
        // TODO
    }

    return refinement;
}
