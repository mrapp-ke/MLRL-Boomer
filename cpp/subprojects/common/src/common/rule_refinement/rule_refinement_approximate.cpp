#include "common/rule_refinement/rule_refinement_approximate.hpp"


template<typename IndexVector, typename RefinementComparator>
static inline void findRefinementInternally(const IndexVector& labelIndices, uint32 featureIndex, bool nominal,
                                            IRuleRefinementCallback<IHistogram, ThresholdVector>& callback,
                                            RefinementComparator& comparator) {
    Refinement refinement;
    refinement.featureIndex = featureIndex;

    // Invoke the callback...
    std::unique_ptr<IRuleRefinementCallback<IHistogram, ThresholdVector>::Result> callbackResultPtr = callback.get();
    const IHistogram& statistics = callbackResultPtr->statistics_;
    const ThresholdVector& thresholdVector = callbackResultPtr->vector_;
    ThresholdVector::const_iterator thresholdIterator = thresholdVector.cbegin();
    uint32 numBins = thresholdVector.getNumElements();
    uint32 sparseBinIndex = thresholdVector.getSparseBinIndex();
    bool sparse = sparseBinIndex < numBins;

    // Create a new, empty subset of the statistics...
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr = statistics.createSubset(labelIndices);

    for (auto it = thresholdVector.missing_indices_cbegin(); it != thresholdVector.missing_indices_cend(); it++) {
        uint32 i = *it;
        statisticsSubsetPtr->addToMissing(i);
    }

    // In the following, we start by processing the bins in range [0, sparseBinIndex)...
    bool subsetModified = false;
    int64 firstR = 0;
    int64 r;

    // Traverse bins in ascending order until the first bin with non-zero weight is encountered...
    for (r = 0; r < sparseBinIndex; r++) {
        if (statisticsSubsetPtr->hasNonZeroWeight(r)) {
            // Add the bin to the subset to mark it as covered by upcoming refinements...
            statisticsSubsetPtr->addToSubset(r);
            subsetModified = true;
            break;
        }
    }

    // Traverse the remaining bins in ascending order...
    if (subsetModified) {
        for (r = r + 1; r < sparseBinIndex; r++) {
            // Do only consider bins that are not empty...
            if (statisticsSubsetPtr->hasNonZeroWeight(r)) {
                // Find and evaluate the best head for the current refinement, if a condition that uses the <= operator
                // (or the == operator in case of a nominal feature) is used...
                const IScoreVector& scoreVector = statisticsSubsetPtr->evaluate();

                // If the refinement is better than the current rule...
                if (comparator.isImprovement(scoreVector)) {
                    refinement.start = firstR;
                    refinement.end = r;
                    refinement.covered = true;
                    refinement.threshold = thresholdIterator[r - 1];
                    refinement.comparator = nominal ? EQ : LEQ;
                    comparator.pushRefinement(refinement, scoreVector);
                }

                // Find and evaluate the best head for the current refinement, if a condition that uses the > operator
                // (or the != operator in case of a nominal feature) is used...
                const IScoreVector& scoreVector2 = statisticsSubsetPtr->evaluateUncovered();

                // If the refinement is better than the current rule...
                if (comparator.isImprovement(scoreVector2)) {
                    refinement.start = firstR;
                    refinement.end = r;
                    refinement.covered = false;
                    refinement.threshold = thresholdIterator[r - 1];
                    refinement.comparator = nominal ? NEQ : GR;
                    comparator.pushRefinement(refinement, scoreVector2);
                }

                // Reset the subset in case of a nominal feature, as the previous bins will not be covered by the next
                // condition...
                if (nominal) {
                    statisticsSubsetPtr->resetSubset();
                    firstR = r;
                }

                // Add the bin to the subset to mark it as covered by upcoming refinements...
                statisticsSubsetPtr->addToSubset(r);
            }
        }

        // If any bins have been processed so far and if there is a sparse bin, we must evaluate additional conditions
        // that separate the bins that have been iterated from the remaining ones (including the sparse bin)...
        if (subsetModified && sparse) {
            // Find and evaluate the best head for the current refinement, if a condition that uses the <= operator (or
            // the == operator in case of nominal feature) is used...
            const IScoreVector& scoreVector = statisticsSubsetPtr->evaluate();

            // If the refinement is better than the current rule...
            if (comparator.isImprovement(scoreVector)) {
                refinement.start = firstR;
                refinement.end = sparseBinIndex;
                refinement.covered = true;
                refinement.threshold = thresholdIterator[sparseBinIndex - 1];
                refinement.comparator = nominal ? EQ : LEQ;
                comparator.pushRefinement(refinement, scoreVector);
            }

            // Find and evaluate the best head for the current refinement, if a condition that uses the > operator (or
            // the != operator in case of a nominal feature) is used...
            const IScoreVector& scoreVector2 = statisticsSubsetPtr->evaluateUncovered();

            // If the refinement is better than the current rule...
            if (comparator.isImprovement(scoreVector2)) {
                refinement.start = firstR;
                refinement.end = sparseBinIndex;
                refinement.covered = false;
                refinement.threshold = thresholdIterator[sparseBinIndex - 1];
                refinement.comparator = nominal ? NEQ : GR;
                comparator.pushRefinement(refinement, scoreVector2);
            }
        }

        // Reset the subset, if any bins have been processed...
        statisticsSubsetPtr->resetSubset();
    }

    bool subsetModifiedPrevious = subsetModified;

    // We continue by processing the bins in range (sparseBinIndex, numBins)...
    subsetModified = false;
    firstR = ((int64) numBins) - 1;

    // Traverse bins in descending order until the first bin with non-zero weight is encountered...
    for (r = firstR; r > sparseBinIndex; r--) {
        if (statisticsSubsetPtr->hasNonZeroWeight(r)) {
            // Add the bin to the subset to mark it as covered by upcoming refinements...
            statisticsSubsetPtr->addToSubset(r);
            subsetModified = true;
            break;
        }
    }

    // Traverse the remaining bins in descending order...
    if (subsetModified) {
        for (r = r - 1; r > sparseBinIndex; r--) {
            // Do only consider bins that are not empty...
            if (statisticsSubsetPtr->hasNonZeroWeight(r)) {
                // Find and evaluate the best head for the current refinement, if a condition that uses the > operator
                // (or the == operator in case of a nominal feature) is used..
                const IScoreVector& scoreVector = statisticsSubsetPtr->evaluate();

                // If the refinement is better than the current rule...
                if (comparator.isImprovement(scoreVector)) {
                    refinement.start = firstR;
                    refinement.end = r;
                    refinement.covered = true;

                    if (nominal) {
                        refinement.threshold = thresholdIterator[firstR];
                        refinement.comparator = EQ;
                    } else {
                        refinement.threshold = thresholdIterator[r];
                        refinement.comparator = GR;
                    }

                    comparator.pushRefinement(refinement, scoreVector);
                }

                // Find and evaluate the best head for the current refinement, if a condition that uses the <= operator
                // (or the != operator in case of a nominal feature) is used...
                const IScoreVector& scoreVector2 = statisticsSubsetPtr->evaluateUncovered();

                // If the refinement is better than the current rule...
                if (comparator.isImprovement(scoreVector2)) {
                    refinement.start = firstR;
                    refinement.end = r;
                    refinement.covered = false;

                    if (nominal) {
                        refinement.threshold = thresholdIterator[firstR];
                        refinement.comparator = NEQ;
                    } else {
                        refinement.threshold = thresholdIterator[r];
                        refinement.comparator = LEQ;
                    }

                    comparator.pushRefinement(refinement, scoreVector2);
                }

                // Reset the subset in case of a nominal feature, as the previous bins will not be covered by the next
                // condition...
                if (nominal) {
                    statisticsSubsetPtr->resetSubset();
                    firstR = r;
                }

                // Add the bin to the subset to mark it as covered by upcoming refinements...
                statisticsSubsetPtr->addToSubset(r);
            }
        }

        // If there is a sparse bin, we must evaluate additional conditions that separate the bins in range
        // (sparseBinIndex, numBins) from the remaining ones...
        if (sparse) {
            // Find and evaluate the best head for the current refinement, if
            const IScoreVector& scoreVector = statisticsSubsetPtr->evaluate();

            // If the refinement is better than the current rule...
            if (comparator.isImprovement(scoreVector)) {
                refinement.start = firstR;
                refinement.end = sparseBinIndex;
                refinement.covered = true;

                if (nominal) {
                    refinement.threshold = thresholdIterator[firstR];
                    refinement.comparator = EQ;
                } else {
                    refinement.threshold = thresholdIterator[sparseBinIndex];
                    refinement.comparator = GR;
                }

                comparator.pushRefinement(refinement, scoreVector);
            }

            // Find and evaluate the best head for the current refinement, if
            const IScoreVector& scoreVector2 = statisticsSubsetPtr->evaluateUncovered();

            // If the refinement is better than the current rule...
            if (comparator.isImprovement(scoreVector2)) {
                refinement.start = firstR;
                refinement.end = sparseBinIndex;
                refinement.covered = false;

                if (nominal) {
                    refinement.threshold = thresholdIterator[firstR];
                    refinement.comparator = NEQ;
                } else {
                    refinement.threshold = thresholdIterator[sparseBinIndex];
                    refinement.comparator = LEQ;
                }

                comparator.pushRefinement(refinement, scoreVector2);
            }

            // If the feature is nominal and if any bins in the range [0, sparseBinIndex) have been processed earlier,
            // we must test additional conditions that separate the sparse bin from the remaining bins...
            if (nominal && subsetModifiedPrevious) {
                // Reset the subset once again to ensure that the accumulated state includes all bins that have been
                // processed so far...
                statisticsSubsetPtr->resetSubset();

                // Find and evaluate the best head for the current refinement, if the condition
                // `f != thresholdIterator[sparseBinIndex]` is used...
                const IScoreVector& scoreVector = statisticsSubsetPtr->evaluateAccumulated();

                // If the refinement is better than the current rule...
                if (comparator.isImprovement(scoreVector)) {
                    refinement.start = sparseBinIndex;
                    refinement.end = sparseBinIndex + 1;
                    refinement.covered = false;
                    refinement.threshold = thresholdIterator[sparseBinIndex];
                    refinement.comparator = NEQ;
                    comparator.pushRefinement(refinement, scoreVector);
                }

                // Find and evaluate the best head for the current refinement, if the condition
                // `f == thresholdIterator[sparseBinIndex]` is used...
                const IScoreVector& scoreVector2 = statisticsSubsetPtr->evaluateUncoveredAccumulated();

                // If the refinement is better than the current rule...
                if (comparator.isImprovement(scoreVector2)) {
                    refinement.start = sparseBinIndex;
                    refinement.end = sparseBinIndex + 1;
                    refinement.covered = true;
                    refinement.threshold = thresholdIterator[sparseBinIndex];
                    refinement.comparator = EQ;
                    comparator.pushRefinement(refinement, scoreVector2);
                }
            }
        }
    }
}

template<typename T>
ApproximateRuleRefinement<T>::ApproximateRuleRefinement(const T& labelIndices, uint32 featureIndex, bool nominal,
                                                        std::unique_ptr<Callback> callbackPtr)
    : labelIndices_(labelIndices), featureIndex_(featureIndex), nominal_(nominal),
      callbackPtr_(std::move(callbackPtr)) {

}

template<typename T>
void ApproximateRuleRefinement<T>::findRefinement(SingleRefinementComparator& comparator) {
    findRefinementInternally(labelIndices_, featureIndex_, nominal_, *callbackPtr_, comparator);
}

template class ApproximateRuleRefinement<CompleteIndexVector>;
template class ApproximateRuleRefinement<PartialIndexVector>;
