#include "common/rule_refinement/rule_refinement_approximate.hpp"


template<class T>
ApproximateRuleRefinement<T>::ApproximateRuleRefinement(
        std::unique_ptr<IHeadRefinement> headRefinementPtr, const T& labelIndices, uint32 featureIndex, bool nominal,
        std::unique_ptr<IRuleRefinementCallback<ThresholdVector, BinWeightVector>> callbackPtr)
    : headRefinementPtr_(std::move(headRefinementPtr)), labelIndices_(labelIndices), featureIndex_(featureIndex),
      nominal_(nominal), callbackPtr_(std::move(callbackPtr)) {

}

template<class T>
void ApproximateRuleRefinement<T>::findRefinement(const AbstractEvaluatedPrediction* currentHead) {
    std::unique_ptr<Refinement> refinementPtr = std::make_unique<Refinement>();
    refinementPtr->featureIndex = featureIndex_;
    refinementPtr->start = 0;
    const AbstractEvaluatedPrediction* bestHead = currentHead;

    // Invoke the callback...
    std::unique_ptr<IRuleRefinementCallback<ThresholdVector, BinWeightVector>::Result> callbackResultPtr =
        callbackPtr_->get();
    const IImmutableStatistics& statistics = callbackResultPtr->statistics_;
    const BinWeightVector& weights = callbackResultPtr->weights_;
    BinWeightVector::const_iterator weightIterator = weights.cbegin();
    const ThresholdVector& thresholdVector = callbackResultPtr->vector_;
    ThresholdVector::const_iterator thresholdIterator = thresholdVector.cbegin();
    uint32 numBins = thresholdVector.getNumElements();

    // Create a new, empty subset of the current statistics when processing a new feature...
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = labelIndices_.createSubset(statistics);

    // Search for the first non-empty bin...
    uint32 r = 0;
    float32 threshold = 0;

    for (; r < numBins; r++) {
        uint32 weight = weightIterator[r];

        if (weight > 0) {
            threshold = thresholdIterator[r];
            statisticsSubsetPtr->addToSubset(r, 1);
            break;
        }
    }

    for (r = r + 1; r < numBins; r++) {
        uint32 weight = weightIterator[r];

        if (weight > 0) {
            const AbstractEvaluatedPrediction* head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr,
                                                                                   false, false);

            if (head != nullptr) {
                bestHead = head;
                refinementPtr->end = r;
                refinementPtr->covered = true;
                refinementPtr->threshold = threshold;
                refinementPtr->comparator = nominal_ ? EQ : LEQ;
            }

            head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr, true, false);

            if (head != nullptr) {
                bestHead = head;
                refinementPtr->end = r;
                refinementPtr->covered = false;
                refinementPtr->threshold = threshold;
                refinementPtr->comparator = nominal_ ? NEQ : GR;
            }

            threshold = thresholdIterator[r];
            statisticsSubsetPtr->addToSubset(r, 1);

            // Reset the subset in case of a nominal feature, as the previous bins will not be covered by the next
            // condition...
            if (nominal_) {
                statisticsSubsetPtr->resetSubset();
            }
        }
    }

    refinementPtr->headPtr = headRefinementPtr_->pollHead();
    refinementPtr_ = std::move(refinementPtr);
}

template<class T>
std::unique_ptr<Refinement> ApproximateRuleRefinement<T>::pollRefinement() {
    return std::move(refinementPtr_);
}

template class ApproximateRuleRefinement<FullIndexVector>;
template class ApproximateRuleRefinement<PartialIndexVector>;
