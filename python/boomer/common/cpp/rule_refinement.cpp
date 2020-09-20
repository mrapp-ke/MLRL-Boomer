#include "rule_refinement.h"
#include <math.h>
#include <memory>

AbstractRuleRefinement::~AbstractRuleRefinement() {

}

Refinement AbstractRuleRefinement::findRefinement(AbstractHeadRefinement* headRefinement,
                                                  PredictionCandidate* currentHead, uint32 numLabelIndices,
                                                  const uint32* labelIndices) {
    Refinement refinement;
    return refinement;
}

ExactRuleRefinementImpl::ExactRuleRefinementImpl(AbstractStatistics* statistics,
                                                 IndexedFloat32ArrayWrapper* indexedArrayWrapper,
                                                 IndexedFloat32Array* indexedArray, const uint32* weights,
                                                 uint32 totalSumOfWeights, uint32 featureIndex, bool nominal) {
    statistics_ = statistics;
    indexedArrayWrapper_ = indexedArrayWrapper;
    indexedArray_ = indexedArray;
    weights_ = weights;
    totalSumOfWeights_ = totalSumOfWeights;
    featureIndex_ = featureIndex;
    nominal_ = nominal;
}

ExactRuleRefinementImpl::~ExactRuleRefinementImpl() {

}

Refinement ExactRuleRefinementImpl::findRefinement(AbstractHeadRefinement* headRefinement,
                                                   PredictionCandidate* currentHead, uint32 numLabelIndices,
                                                   const uint32* labelIndices) {
    // The current refinement of the existing rule
    Refinement refinement;
    refinement.featureIndex = featureIndex_;
    refinement.head = NULL;
    refinement.indexedArray = indexedArray_;
    refinement.indexedArrayWrapper = indexedArrayWrapper_;
    // The best head seen so far
    PredictionCandidate* bestHead = currentHead;
    // Create a new, empty subset of the current statistics when processing a new feature...
    std::unique_ptr<AbstractStatisticsSubset> statisticsSubsetPtr;
    statisticsSubsetPtr.reset(statistics_->createSubset(numLabelIndices, labelIndices));
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
            // Add the example to the subset to mark it as covered by upcoming refinements...
            statisticsSubsetPtr.get()->addToSubset(i, weight);
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
                    // Find and evaluate the best head for the current refinement, if a condition that uses the <=
                    // operator (or the == operator in case of a nominal feature) is used...
                    PredictionCandidate* currentHead = headRefinement->findHead(bestHead, refinement.head, labelIndices,
                                                                                statisticsSubsetPtr.get(), false,
                                                                                false);

                    // If the refinement is better than the current rule...
                    if (currentHead != NULL) {
                        bestHead = currentHead;
                        refinement.head = currentHead;
                        refinement.start = firstR;
                        refinement.end = r;
                        refinement.previous = previousR;
                        refinement.coveredWeights = sumOfWeights;
                        refinement.covered = true;

                        if (nominal_) {
                            refinement.comparator = EQ;
                            refinement.threshold = previousThreshold;
                        } else {
                            refinement.comparator = LEQ;
                            refinement.threshold = (previousThreshold + currentThreshold) / 2.0;
                        }
                    }

                    // Find and evaluate the best head for the current refinement, if a condition that uses the >
                    // operator (or the != operator in case of a nominal feature) is used...
                    currentHead = headRefinement->findHead(bestHead, refinement.head, labelIndices,
                                                           statisticsSubsetPtr.get(), true, false);

                    // If the refinement is better than the current rule...
                    if (currentHead != NULL) {
                        bestHead = currentHead;
                        refinement.head = currentHead;
                        refinement.start = firstR;
                        refinement.end = r;
                        refinement.previous = previousR;
                        refinement.coveredWeights = (totalSumOfWeights_ - sumOfWeights);
                        refinement.covered = false;

                        if (nominal_) {
                            refinement.comparator = NEQ;
                            refinement.threshold = previousThreshold;
                        } else {
                            refinement.comparator = GR;
                            refinement.threshold = (previousThreshold + currentThreshold) / 2.0;
                        }
                    }

                    // Reset the subset in case of a nominal feature, as the previous examples will not be covered by
                    // the next condition...
                    if (nominal_) {
                        statisticsSubsetPtr.get()->resetSubset();
                        sumOfWeights = 0;
                        firstR = r;
                    }
                }

                previousThreshold = currentThreshold;
                previousR = r;

                // Add the example to the subset to mark it as covered by upcoming refinements...
                statisticsSubsetPtr.get()->addToSubset(i, weight);
                sumOfWeights += weight;
                accumulatedSumOfWeights += weight;
            }
        }

        // If the feature is nominal and the examples that have been iterated so far do not all have the same feature
        // value, or if not all examples have been iterated so far, we must evaluate additional conditions
        // `f == previous_threshold` and `f != previous_threshold`...
        if (nominal_ && sumOfWeights > 0 && (sumOfWeights < accumulatedSumOfWeights
                                             || accumulatedSumOfWeights < totalSumOfWeights_)) {
            // Find and evaluate the best head for the current refinement, if a condition that uses the == operator is
            // used...
            PredictionCandidate* currentHead = headRefinement->findHead(bestHead, refinement.head, labelIndices,
                                                                        statisticsSubsetPtr.get(), false, false);

            if (currentHead != NULL) {
                bestHead = currentHead;
                refinement.head = currentHead;
                refinement.start = firstR;
                refinement.end = (lastNegativeR + 1);
                refinement.previous = previousR;
                refinement.coveredWeights = sumOfWeights;
                refinement.covered = true;
                refinement.comparator = EQ;
                refinement.threshold = previousThreshold;
            }

            // Find and evaluate the best head for the current refinement, if a condition that uses the != operator is
            // used...
            currentHead = headRefinement->findHead(bestHead, refinement.head, labelIndices, statisticsSubsetPtr.get(),
                                                   true, false);

            if (currentHead != NULL) {
                bestHead = currentHead;
                refinement.head = currentHead;
                refinement.start = firstR;
                refinement.end = (lastNegativeR + 1);
                refinement.previous = previousR;
                refinement.coveredWeights = (totalSumOfWeights_ - sumOfWeights);
                refinement.covered = false;
                refinement.comparator = NEQ;
                refinement.threshold = previousThreshold;
            }
        }

        // Reset the subset, if any examples with feature value < 0 have been processed...
        statisticsSubsetPtr.get()->resetSubset();
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
            // Add the example to the subset to mark it as covered by upcoming refinements...
            statisticsSubsetPtr.get()->addToSubset(i, weight);
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
                    // Find and evaluate the best head for the current refinement, if a condition that uses the >
                    // operator (or the == operator in case of a nominal feature) is used...
                    PredictionCandidate* currentHead = headRefinement->findHead(bestHead, refinement.head, labelIndices,
                                                                                statisticsSubsetPtr.get(), false,
                                                                                false);

                    // If the refinement is better than the current rule...
                    if (currentHead != NULL) {
                        bestHead = currentHead;
                        refinement.head = currentHead;
                        refinement.start = firstR;
                        refinement.end = r;
                        refinement.previous = previousR;
                        refinement.coveredWeights = sumOfWeights;
                        refinement.covered = true;

                        if (nominal_) {
                            refinement.comparator = EQ;
                            refinement.threshold = previousThreshold;
                        } else {
                            refinement.comparator = GR;
                            refinement.threshold = (previousThreshold + currentThreshold) / 2.0;
                        }
                    }

                    // Find and evaluate the best head for the current refinement, if a condition that uses the <=
                    // operator (or the != operator in case of a nominal feature) is used...
                    currentHead = headRefinement->findHead(bestHead, refinement.head, labelIndices,
                                                           statisticsSubsetPtr.get(), true, false);

                    // If the refinement is better than the current rule...
                    if (currentHead != NULL) {
                        bestHead = currentHead;
                        refinement.head = currentHead;
                        refinement.start = firstR;
                        refinement.end = r;
                        refinement.previous = previousR;
                        refinement.coveredWeights = (totalSumOfWeights_ - sumOfWeights);
                        refinement.covered = false;

                        if (nominal_) {
                            refinement.comparator = NEQ;
                            refinement.threshold = previousThreshold;
                        } else {
                            refinement.comparator = LEQ;
                            refinement.threshold = (previousThreshold + currentThreshold) / 2.0;
                        }
                    }

                    // Reset the subet in case of a nominal feature, as the previous examples will not be covered by
                    // the next condition...
                    if (nominal_) {
                        statisticsSubsetPtr.get()->resetSubset();
                        sumOfWeights = 0;
                        firstR = r;
                    }
                }

                previousThreshold = currentThreshold;
                previousR = r;

                // Add the example to the subset to mark it as covered by upcoming refinements...
                statisticsSubsetPtr.get()->addToSubset(i, weight);
                sumOfWeights += weight;
                accumulatedSumOfWeights += weight;
            }
        }
    }

    // If the feature is nominal and the examples with feature values >= 0 that have been iterated so far do not all
    // have the same feature value, we must evaluate additional conditions `f == previous_threshold` and
    // `f != previous_threshold`...
    if (nominal_ && sumOfWeights > 0 && sumOfWeights < accumulatedSumOfWeights) {
        // Find and evaluate the best head for the current refinement, if a condition that uses the == operator is
        // used...
        PredictionCandidate* currentHead = headRefinement->findHead(bestHead, refinement.head, labelIndices,
                                                                    statisticsSubsetPtr.get(), false, false);

        // If the refinement is better than the current rule...
        if (currentHead != NULL) {
            bestHead = currentHead;
            refinement.head = currentHead;
            refinement.start = firstR;
            refinement.end = lastNegativeR;
            refinement.previous = previousR;
            refinement.coveredWeights = sumOfWeights;
            refinement.covered = true;
            refinement.comparator = EQ;
            refinement.threshold = previousThreshold;
        }

        // Find and evaluate the best head for the current refinement, if a condition that uses the != operator is
        // used...
        currentHead = headRefinement->findHead(bestHead, refinement.head, labelIndices, statisticsSubsetPtr.get(), true,
                                               false);

        // If the refinement is better than the current rule...
        if (currentHead != NULL) {
            bestHead = currentHead;
            refinement.head = currentHead;
            refinement.start = firstR;
            refinement.end = lastNegativeR;
            refinement.previous = previousR;
            refinement.coveredWeights = (totalSumOfWeights_ - sumOfWeights);
            refinement.covered = false;
            refinement.comparator = NEQ;
            refinement.threshold = previousThreshold;
        }
    }

    uint32 totalAccumulatedSumOfWeights = accumulatedSumOfWeightsNegative + accumulatedSumOfWeights;

    // If the sum of weights of all examples that have been iterated so far (including those with feature values < 0 and
    // those with feature values >= 0) is less than the sum of of weights of all examples, this means that there are
    // examples with sparse, i.e. zero, feature values. In such case, we must explicitly test conditions that separate
    // these examples from the ones that have already been iterated...
    if (totalAccumulatedSumOfWeights > 0 && totalAccumulatedSumOfWeights < totalSumOfWeights_) {
        // If the feature is nominal, we must reset the subset once again to ensure that the accumulated state includes
        // all examples that have been processed so far...
        if (nominal_) {
            statisticsSubsetPtr.get()->resetSubset();
            firstR = ((intp) numIndexedValues) - 1;
        }

        // Find and evaluate the best head for the current refinement, if the condition `f > previous_threshold / 2` (or
        // the condition `f != 0` in case of a nominal feature) is used...
        PredictionCandidate* currentHead = headRefinement->findHead(bestHead, refinement.head, labelIndices,
                                                                    statisticsSubsetPtr.get(), false, nominal_);

        // If the refinement is better than the current rule...
        if (currentHead != NULL) {
            bestHead = currentHead;
            refinement.head = currentHead;
            refinement.start = firstR;
            refinement.covered = true;

            if (nominal_) {
                refinement.end = -1;
                refinement.previous = -1;
                refinement.coveredWeights = totalAccumulatedSumOfWeights;
                refinement.comparator = NEQ;
                refinement.threshold = 0.0;
            } else {
                refinement.end = lastNegativeR;
                refinement.previous = previousR;
                refinement.coveredWeights = accumulatedSumOfWeights;
                refinement.comparator = GR;
                refinement.threshold = previousThreshold / 2.0;
            }
        }

        // Find and evaluate the best head for the current refinement, if the condition `f <= previous_threshold / 2`
        // (or `f == 0` in case of a nominal feature) is used...
        currentHead = headRefinement->findHead(bestHead, refinement.head, labelIndices, statisticsSubsetPtr.get(), true,
                                               nominal_);

        // If the refinement is better than the current rule...
        if (currentHead != NULL) {
            bestHead = currentHead;
            refinement.head = currentHead;
            refinement.start = firstR;
            refinement.covered = false;

            if (nominal_) {
                refinement.end = -1;
                refinement.previous = -1;
                refinement.coveredWeights = (totalSumOfWeights_ - totalAccumulatedSumOfWeights);
                refinement.comparator = EQ;
                refinement.threshold = 0.0;
            } else {
                refinement.end = lastNegativeR;
                refinement.previous = previousR;
                refinement.coveredWeights = (totalSumOfWeights_ - accumulatedSumOfWeights);
                refinement.comparator = LEQ;
                refinement.threshold = previousThreshold / 2.0;
            }
        }
    }

    // If the feature is numerical and there are other examples than those with feature values < 0 that have been
    // processed earlier, we must evaluate additional conditions that separate the examples with feature values < 0 from
    // the remaining ones (unlike in the nominal case, these conditions cannot be evaluated earlier, because it remains
    // unclear what the thresholds of the conditions should be until the examples with feature values >= 0 have been
    // processed).
    if (!nominal_ && accumulatedSumOfWeightsNegative > 0 && accumulatedSumOfWeightsNegative < totalSumOfWeights_) {
        // Find and evaluate the best head for the current refinement, if the condition that uses the <= operator is
        // used...
        PredictionCandidate* currentHead = headRefinement->findHead(bestHead, refinement.head, labelIndices,
                                                                    statisticsSubsetPtr.get(), false, true);

        if (currentHead != NULL) {
            bestHead = currentHead;
            refinement.head = currentHead;
            refinement.start = 0;
            refinement.end = (lastNegativeR + 1);
            refinement.previous = previousRNegative;
            refinement.coveredWeights = accumulatedSumOfWeightsNegative;
            refinement.covered = true;
            refinement.comparator = LEQ;

            if (totalAccumulatedSumOfWeights < totalSumOfWeights_) {
                // If the condition separates an example with feature value < 0 from an (sparse) example with feature
                // value == 0
                refinement.threshold = previousThresholdNegative / 2.0;
            } else {
                // If the condition separates an examples with feature value < 0 from an example with feature value > 0
                refinement.threshold = previousThresholdNegative
                                       + (fabs(previousThreshold - previousThresholdNegative) / 2.0);
            }
        }

        // Find and evaluate the best head for the current refinement, if the condition that uses the > operator is
        // used...
        currentHead = headRefinement->findHead(bestHead, refinement.head, labelIndices, statisticsSubsetPtr.get(), true,
                                               true);

        if (currentHead != NULL) {
            bestHead = currentHead;
            refinement.head = currentHead;
            refinement.start = 0;
            refinement.end = (lastNegativeR + 1);
            refinement.previous = previousRNegative;
            refinement.coveredWeights = (totalSumOfWeights_ - accumulatedSumOfWeightsNegative);
            refinement.covered = false;
            refinement.comparator = GR;

            if (totalAccumulatedSumOfWeights < totalSumOfWeights_) {
                // If the condition separates an example with feature value < 0 from an (sparse) example with feature
                // value == 0
                refinement.threshold = previousThresholdNegative / 2.0;
            } else {
                // If the condition separates an examples with feature value < 0 from an example with feature value > 0
                refinement.threshold = previousThresholdNegative
                                       + (fabs(previousThreshold - previousThresholdNegative) / 2.0);
            }
        }
    }

    return refinement;
}
