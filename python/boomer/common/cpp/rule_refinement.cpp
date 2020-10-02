#include "rule_refinement.h"
#include <math.h>
#include <memory>

ExactRuleRefinementImpl::ExactRuleRefinementImpl(AbstractStatistics* statistics, IWeightVector* weights,
                                                 uint32 totalSumOfWeights, uint32 featureIndex, bool nominal,
                                                 IRuleRefinementCallback<IndexedFloat32Array>* callback) {
    statistics_ = statistics;
    weights_ = weights;
    totalSumOfWeights_ = totalSumOfWeights;
    featureIndex_ = featureIndex;
    nominal_ = nominal;
    callback_ = callback;
    bestRefinement_.featureIndex = featureIndex;
    bestRefinement_.head = NULL;
}

ExactRuleRefinementImpl::~ExactRuleRefinementImpl() {
    delete callback_;
}

void ExactRuleRefinementImpl::findRefinement(IHeadRefinement* headRefinement, PredictionCandidate* currentHead,
                                             uint32 numLabelIndices, const uint32* labelIndices) {
    // The best head seen so far
    PredictionCandidate* bestHead = currentHead;
    // Create a new, empty subset of the current statistics when processing a new feature...
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr;
    statisticsSubsetPtr.reset(statistics_->createSubset(numLabelIndices, labelIndices));

    // Retrieve the array to be iterated...
    IndexedFloat32Array* indexedArray = callback_->get(featureIndex_);
    IndexedFloat32* indexedValues = indexedArray->data;
    uint32 numIndexedValues = indexedArray->numElements;

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
        uint32 weight = weights_->getValue(i);

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
            uint32 weight = weights_->getValue(i);

            // Do only consider examples that are included in the current sub-sample...
            if (weight > 0) {
                // Split points between examples with the same feature value must not be considered...
                if (previousThreshold != currentThreshold) {
                    // Find and evaluate the best head for the current refinement, if a condition that uses the <=
                    // operator (or the == operator in case of a nominal feature) is used...
                    PredictionCandidate* currentHead = headRefinement->findHead(bestHead, bestRefinement_.head,
                                                                                labelIndices, statisticsSubsetPtr.get(),
                                                                                false, false);

                    // If the refinement is better than the current rule...
                    if (currentHead != NULL) {
                        bestHead = currentHead;
                        bestRefinement_.head = currentHead;
                        bestRefinement_.start = firstR;
                        bestRefinement_.end = r;
                        bestRefinement_.previous = previousR;
                        bestRefinement_.coveredWeights = sumOfWeights;
                        bestRefinement_.covered = true;

                        if (nominal_) {
                            bestRefinement_.comparator = EQ;
                            bestRefinement_.threshold = previousThreshold;
                        } else {
                            bestRefinement_.comparator = LEQ;
                            bestRefinement_.threshold = (previousThreshold + currentThreshold) / 2.0;
                        }
                    }

                    // Find and evaluate the best head for the current refinement, if a condition that uses the >
                    // operator (or the != operator in case of a nominal feature) is used...
                    currentHead = headRefinement->findHead(bestHead, bestRefinement_.head, labelIndices,
                                                           statisticsSubsetPtr.get(), true, false);

                    // If the refinement is better than the current rule...
                    if (currentHead != NULL) {
                        bestHead = currentHead;
                        bestRefinement_.head = currentHead;
                        bestRefinement_.start = firstR;
                        bestRefinement_.end = r;
                        bestRefinement_.previous = previousR;
                        bestRefinement_.coveredWeights = (totalSumOfWeights_ - sumOfWeights);
                        bestRefinement_.covered = false;

                        if (nominal_) {
                            bestRefinement_.comparator = NEQ;
                            bestRefinement_.threshold = previousThreshold;
                        } else {
                            bestRefinement_.comparator = GR;
                            bestRefinement_.threshold = (previousThreshold + currentThreshold) / 2.0;
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
            PredictionCandidate* currentHead = headRefinement->findHead(bestHead, bestRefinement_.head, labelIndices,
                                                                        statisticsSubsetPtr.get(), false, false);

            if (currentHead != NULL) {
                bestHead = currentHead;
                bestRefinement_.head = currentHead;
                bestRefinement_.start = firstR;
                bestRefinement_.end = (lastNegativeR + 1);
                bestRefinement_.previous = previousR;
                bestRefinement_.coveredWeights = sumOfWeights;
                bestRefinement_.covered = true;
                bestRefinement_.comparator = EQ;
                bestRefinement_.threshold = previousThreshold;
            }

            // Find and evaluate the best head for the current refinement, if a condition that uses the != operator is
            // used...
            currentHead = headRefinement->findHead(bestHead, bestRefinement_.head, labelIndices,
                                                   statisticsSubsetPtr.get(), true, false);

            if (currentHead != NULL) {
                bestHead = currentHead;
                bestRefinement_.head = currentHead;
                bestRefinement_.start = firstR;
                bestRefinement_.end = (lastNegativeR + 1);
                bestRefinement_.previous = previousR;
                bestRefinement_.coveredWeights = (totalSumOfWeights_ - sumOfWeights);
                bestRefinement_.covered = false;
                bestRefinement_.comparator = NEQ;
                bestRefinement_.threshold = previousThreshold;
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
        uint32 weight = weights_->getValue(i);

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
            uint32 weight = weights_->getValue(i);

            // Do only consider examples that are included in the current sub-sample...
            if (weight > 0) {
                float32 currentThreshold = indexedValues[r].value;

                // Split points between examples with the same feature value must not be considered...
                if (previousThreshold != currentThreshold) {
                    // Find and evaluate the best head for the current refinement, if a condition that uses the >
                    // operator (or the == operator in case of a nominal feature) is used...
                    PredictionCandidate* currentHead = headRefinement->findHead(bestHead, bestRefinement_.head,
                                                                                labelIndices, statisticsSubsetPtr.get(),
                                                                                false, false);

                    // If the refinement is better than the current rule...
                    if (currentHead != NULL) {
                        bestHead = currentHead;
                        bestRefinement_.head = currentHead;
                        bestRefinement_.start = firstR;
                        bestRefinement_.end = r;
                        bestRefinement_.previous = previousR;
                        bestRefinement_.coveredWeights = sumOfWeights;
                        bestRefinement_.covered = true;

                        if (nominal_) {
                            bestRefinement_.comparator = EQ;
                            bestRefinement_.threshold = previousThreshold;
                        } else {
                            bestRefinement_.comparator = GR;
                            bestRefinement_.threshold = (previousThreshold + currentThreshold) / 2.0;
                        }
                    }

                    // Find and evaluate the best head for the current refinement, if a condition that uses the <=
                    // operator (or the != operator in case of a nominal feature) is used...
                    currentHead = headRefinement->findHead(bestHead, bestRefinement_.head, labelIndices,
                                                           statisticsSubsetPtr.get(), true, false);

                    // If the refinement is better than the current rule...
                    if (currentHead != NULL) {
                        bestHead = currentHead;
                        bestRefinement_.head = currentHead;
                        bestRefinement_.start = firstR;
                        bestRefinement_.end = r;
                        bestRefinement_.previous = previousR;
                        bestRefinement_.coveredWeights = (totalSumOfWeights_ - sumOfWeights);
                        bestRefinement_.covered = false;

                        if (nominal_) {
                            bestRefinement_.comparator = NEQ;
                            bestRefinement_.threshold = previousThreshold;
                        } else {
                            bestRefinement_.comparator = LEQ;
                            bestRefinement_.threshold = (previousThreshold + currentThreshold) / 2.0;
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
    }

    // If the feature is nominal and the examples with feature values >= 0 that have been iterated so far do not all
    // have the same feature value, we must evaluate additional conditions `f == previous_threshold` and
    // `f != previous_threshold`...
    if (nominal_ && sumOfWeights > 0 && sumOfWeights < accumulatedSumOfWeights) {
        // Find and evaluate the best head for the current refinement, if a condition that uses the == operator is
        // used...
        PredictionCandidate* currentHead = headRefinement->findHead(bestHead, bestRefinement_.head, labelIndices,
                                                                    statisticsSubsetPtr.get(), false, false);

        // If the refinement is better than the current rule...
        if (currentHead != NULL) {
            bestHead = currentHead;
            bestRefinement_.head = currentHead;
            bestRefinement_.start = firstR;
            bestRefinement_.end = lastNegativeR;
            bestRefinement_.previous = previousR;
            bestRefinement_.coveredWeights = sumOfWeights;
            bestRefinement_.covered = true;
            bestRefinement_.comparator = EQ;
            bestRefinement_.threshold = previousThreshold;
        }

        // Find and evaluate the best head for the current refinement, if a condition that uses the != operator is
        // used...
        currentHead = headRefinement->findHead(bestHead, bestRefinement_.head, labelIndices, statisticsSubsetPtr.get(),
                                               true, false);

        // If the refinement is better than the current rule...
        if (currentHead != NULL) {
            bestHead = currentHead;
            bestRefinement_.head = currentHead;
            bestRefinement_.start = firstR;
            bestRefinement_.end = lastNegativeR;
            bestRefinement_.previous = previousR;
            bestRefinement_.coveredWeights = (totalSumOfWeights_ - sumOfWeights);
            bestRefinement_.covered = false;
            bestRefinement_.comparator = NEQ;
            bestRefinement_.threshold = previousThreshold;
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
        PredictionCandidate* currentHead = headRefinement->findHead(bestHead, bestRefinement_.head, labelIndices,
                                                                    statisticsSubsetPtr.get(), false, nominal_);

        // If the refinement is better than the current rule...
        if (currentHead != NULL) {
            bestHead = currentHead;
            bestRefinement_.head = currentHead;
            bestRefinement_.start = firstR;
            bestRefinement_.covered = true;

            if (nominal_) {
                bestRefinement_.end = -1;
                bestRefinement_.previous = -1;
                bestRefinement_.coveredWeights = totalAccumulatedSumOfWeights;
                bestRefinement_.comparator = NEQ;
                bestRefinement_.threshold = 0.0;
            } else {
                bestRefinement_.end = lastNegativeR;
                bestRefinement_.previous = previousR;
                bestRefinement_.coveredWeights = accumulatedSumOfWeights;
                bestRefinement_.comparator = GR;
                bestRefinement_.threshold = previousThreshold / 2.0;
            }
        }

        // Find and evaluate the best head for the current refinement, if the condition `f <= previous_threshold / 2`
        // (or `f == 0` in case of a nominal feature) is used...
        currentHead = headRefinement->findHead(bestHead, bestRefinement_.head, labelIndices, statisticsSubsetPtr.get(),
                                               true, nominal_);

        // If the refinement is better than the current rule...
        if (currentHead != NULL) {
            bestHead = currentHead;
            bestRefinement_.head = currentHead;
            bestRefinement_.start = firstR;
            bestRefinement_.covered = false;

            if (nominal_) {
                bestRefinement_.end = -1;
                bestRefinement_.previous = -1;
                bestRefinement_.coveredWeights = (totalSumOfWeights_ - totalAccumulatedSumOfWeights);
                bestRefinement_.comparator = EQ;
                bestRefinement_.threshold = 0.0;
            } else {
                bestRefinement_.end = lastNegativeR;
                bestRefinement_.previous = previousR;
                bestRefinement_.coveredWeights = (totalSumOfWeights_ - accumulatedSumOfWeights);
                bestRefinement_.comparator = LEQ;
                bestRefinement_.threshold = previousThreshold / 2.0;
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
        PredictionCandidate* currentHead = headRefinement->findHead(bestHead, bestRefinement_.head, labelIndices,
                                                                    statisticsSubsetPtr.get(), false, true);

        if (currentHead != NULL) {
            bestHead = currentHead;
            bestRefinement_.head = currentHead;
            bestRefinement_.start = 0;
            bestRefinement_.end = (lastNegativeR + 1);
            bestRefinement_.previous = previousRNegative;
            bestRefinement_.coveredWeights = accumulatedSumOfWeightsNegative;
            bestRefinement_.covered = true;
            bestRefinement_.comparator = LEQ;

            if (totalAccumulatedSumOfWeights < totalSumOfWeights_) {
                // If the condition separates an example with feature value < 0 from an (sparse) example with feature
                // value == 0
                bestRefinement_.threshold = previousThresholdNegative / 2.0;
            } else {
                // If the condition separates an examples with feature value < 0 from an example with feature value > 0
                bestRefinement_.threshold = previousThresholdNegative
                                            + (fabs(previousThreshold - previousThresholdNegative) / 2.0);
            }
        }

        // Find and evaluate the best head for the current refinement, if the condition that uses the > operator is
        // used...
        currentHead = headRefinement->findHead(bestHead, bestRefinement_.head, labelIndices, statisticsSubsetPtr.get(),
                                               true, true);

        if (currentHead != NULL) {
            bestHead = currentHead;
            bestRefinement_.head = currentHead;
            bestRefinement_.start = 0;
            bestRefinement_.end = (lastNegativeR + 1);
            bestRefinement_.previous = previousRNegative;
            bestRefinement_.coveredWeights = (totalSumOfWeights_ - accumulatedSumOfWeightsNegative);
            bestRefinement_.covered = false;
            bestRefinement_.comparator = GR;

            if (totalAccumulatedSumOfWeights < totalSumOfWeights_) {
                // If the condition separates an example with feature value < 0 from an (sparse) example with feature
                // value == 0
                bestRefinement_.threshold = previousThresholdNegative / 2.0;
            } else {
                // If the condition separates an examples with feature value < 0 from an example with feature value > 0
                bestRefinement_.threshold = previousThresholdNegative
                                            + (fabs(previousThreshold - previousThresholdNegative) / 2.0);
            }
        }
    }
}

ApproximateRuleRefinementImpl::ApproximateRuleRefinementImpl(AbstractStatistics* statistics, BinArray* binArray,
                                                             uint32 featureIndex,
                                                             IRuleRefinementCallback<IndexedFloat32Array>* callback) {
    statistics_ = statistics;
    binArray_ = binArray;
    featureIndex_ = featureIndex;
    callback_ = callback;
}

void ApproximateRuleRefinementImpl::findRefinement(IHeadRefinement* headRefinement,
                                                         PredictionCandidate* currentHead,
                                                         uint32 numLabelIndices, const uint32* labelIndices) {
    uint32 numBins = binArray_->numBins;
    Refinement refinement;
    refinement.featureIndex = featureIndex_;
    refinement.head = NULL;
    refinement.start = 0;

    PredictionCandidate* bestHead = currentHead;

    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr;
    statisticsSubsetPtr.reset(statistics_->createSubset(numLabelIndices, labelIndices));

    uint32 r = 0;
    //Search for the first not empty bin
    while(binArray_->bins[r].numExamples == 0 && r < numBins){
        r++;
    }
    statisticsSubsetPtr.get()->addToSubset(r, 1);
    uint32 previousR = r;
    float32 previousValue = binArray_->bins[r].maxValue;
    uint32 numCoveredExamples = binArray_->bins[r].numExamples;

    r += 1;
    for(; r < numBins; r++){
        uint32 numExamples = binArray_->bins[r].numExamples;

        if(numExamples > 0){
            float32 currentValue = binArray_->bins[r].minValue;

            PredictionCandidate* currentHead = headRefinement->findHead(bestHead, refinement.head, labelIndices,
                                                          statisticsSubsetPtr.get(), false, false);

            if(currentHead != NULL){
                bestHead = currentHead;
                refinement.head = currentHead;
                refinement.comparator = LEQ;
                refinement.threshold = (previousValue + currentValue)/2.0;
                refinement.end = r;
                refinement.previous = previousR;
                refinement.coveredWeights = numCoveredExamples;
                refinement.covered = true;
            }

            currentHead = headRefinement->findHead(bestHead, refinement.head, labelIndices,
                                                          statisticsSubsetPtr.get(), true, false);

            if(currentHead != NULL){
                bestHead = currentHead;
                refinement.head = currentHead;
                refinement.comparator = GR;
                refinement.threshold = (previousValue + currentValue)/2.0;
                refinement.end = r;
                refinement.previous = previousR;
                refinement.coveredWeights = numCoveredExamples;
                refinement.covered = false;
            }
            previousValue = binArray_->bins[r].maxValue;
            previousR = r;
            numCoveredExamples += numExamples;
            statisticsSubsetPtr.get()->addToSubset(r, 1);
        }
    }
    bestRefinement_ = refinement;
}
